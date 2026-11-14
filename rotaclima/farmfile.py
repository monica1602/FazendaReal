"""Dados reais do produtor: arquivo da fazenda (JSON), limites de talhões, pluviômetro e produtividade."""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from rotaclima.crops import CROPS
from rotaclima.presets import TALHAO_COLUMNS

FORMATO = "fazendareal/1"
SACA_KG = 60.0


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def crop_key(name: str) -> str | None:
    """Reconhece a cultura pelo nome ("Soja", "milho 2ª safra", "milho_2s"...)."""
    n = _norm(name)
    if not n:
        return None
    for k, c in CROPS.items():
        if n in (_norm(k), _norm(c.nome), _norm(c.nome.split("(")[0])):
            return k
    for k, c in CROPS.items():
        if _norm(c.nome).startswith(n):
            return k
    return None


def textura_from_argila(argila_pct: float) -> str:
    """Classes texturais simplificadas usadas no ZARC/Embrapa: < 15 % arenosa, 15–35 % média, ≥ 35 % argilosa."""
    if argila_pct < 15:
        return "arenosa"
    return "média" if argila_pct < 35 else "argilosa"


# ----------------------------------------------------------------------------- limites dos talhões

def polygon_stats(poly: list[tuple[float, float]]) -> tuple[float, float, float]:
    """(lat, lon) do centro e área em hectares de um polígono [(lat, lon), ...]."""
    pts = poly[:-1] if len(poly) > 1 and tuple(poly[0]) == tuple(poly[-1]) else poly
    lat = float(np.mean([p[0] for p in pts]))
    lon = float(np.mean([p[1] for p in pts]))
    x = np.radians([p[1] for p in pts]) * 6_371_000 * np.cos(np.radians(lat))
    y = np.radians([p[0] for p in pts]) * 6_371_000
    area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2 / 10_000
    return lat, lon, float(area)


def _ring(coords) -> list[tuple[float, float]]:
    return [(float(c[1]), float(c[0])) for c in coords]


def parse_geojson(gj: dict) -> list[tuple[str, list[tuple[float, float]]]]:
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    out = []
    for i, f in enumerate(feats):
        geom = f.get("geometry") or f
        if geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
        elif geom.get("type") == "MultiPolygon":
            ring = geom["coordinates"][0][0]
        else:
            continue
        props = f.get("properties") or {}
        out.append((str(props.get("nome") or props.get("name") or f"Talhão {i + 1}"), _ring(ring)))
    if not out:
        raise ValueError("nenhum polígono encontrado")
    return out


def parse_kml(text: str | bytes) -> list[tuple[str, list[tuple[float, float]]]]:
    """Polígonos de um KML (Google Earth / Google Maps)."""
    root = ET.fromstring(text)
    out = []
    for i, pm in enumerate(root.iter()):
        if not pm.tag.endswith("Placemark"):
            continue
        name = next((e.text for e in pm.iter() if e.tag.endswith("name") and e.text), None)
        coords = next((e.text for e in pm.iter() if e.tag.endswith("coordinates") and e.text), None)
        if not coords:
            continue
        pts = [tuple(map(float, c.split(",")[:2])) for c in coords.split()]
        if len(pts) >= 3:
            out.append((name or f"Talhão {len(out) + 1}", _ring(pts)))
    if not out:
        raise ValueError("nenhum polígono encontrado")
    return out


# ----------------------------------------------------------------------------- planilhas (CSV)

def _read_table(data: bytes | str) -> pd.DataFrame:
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    df.columns = [_norm(c).replace(" ", "_") for c in df.columns]
    return df


def _number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.where(~s.str.contains(","), s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def _find(df: pd.DataFrame, *starts: str) -> str:
    for c in df.columns:
        if any(c.startswith(s) for s in starts):
            return c
    raise ValueError(f"coluna não encontrada (esperado: {' ou '.join(starts)})")


def read_rain_csv(data: bytes | str) -> pd.DataFrame:
    """Registros de pluviômetro: colunas `data` e `chuva_mm`, diários (dd/mm/aaaa) ou mensais (mm/aaaa)."""
    df = _read_table(data)
    dcol, rcol = _find(df, "data", "date", "dia", "mes"), _find(df, "chuva", "precip", "mm")
    raw = df[dcol].astype(str).str.strip()
    monthly = raw.str.fullmatch(r"\d{1,2}[/-]\d{4}|\d{4}[/-]\d{1,2}").all()
    if monthly:
        parts = raw.str.split(r"[/-]", expand=True)
        year = np.where(parts[0].str.len() == 4, parts[0], parts[1]).astype(int)
        month = np.where(parts[0].str.len() == 4, parts[1], parts[0]).astype(int)
        dates = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
    else:
        iso = raw.str.match(r"\d{4}-")
        dates = pd.to_datetime(raw.where(iso), format="%Y-%m-%d", errors="coerce").fillna(
            pd.to_datetime(raw.where(~iso), format="%d/%m/%Y", errors="coerce"))
    out = pd.DataFrame({"data": dates, "chuva_mm": _number(df[rcol])}).dropna()
    if out.empty:
        raise ValueError("nenhuma linha com data e chuva válidas")
    out["mensal"] = bool(monthly)
    return out.sort_values("data").reset_index(drop=True)


def rain_calibration(records: pd.DataFrame, power_daily_rain: pd.Series) -> dict | None:
    """Compara o pluviômetro com a NASA POWER nos meses completos em comum.

    Retorna o fator multiplicativo (soma local / soma NASA) e a tabela mensal, ou None sem sobreposição.
    """
    if records is None or records.empty:
        return None
    r = records.set_index("data")["chuva_mm"]
    if bool(records["mensal"].iloc[0]):
        local = r.groupby(r.index.to_period("M")).sum()
    else:
        g = r.groupby(r.index.to_period("M"))
        days = g.count()
        local = g.sum()[days >= days.index.days_in_month - 2]
    power = power_daily_rain.groupby(power_daily_rain.index.to_period("M")).agg(["sum", "count"])
    power = power.loc[power["count"] >= power.index.days_in_month - 2, "sum"]
    common = local.index.intersection(power.index)
    if len(common) == 0 or power[common].sum() <= 0:
        return None
    table = pd.DataFrame({"pluviometro": local[common], "nasa_power": power[common]})
    table.index = table.index.to_timestamp()
    return {"fator": float(table["pluviometro"].sum() / table["nasa_power"].sum()), "meses": len(common),
            "tabela": table}


def read_yield_csv(data: bytes | str) -> pd.DataFrame:
    """Produtividade colhida: `talhao`, `safra`, `cultura` e `produtividade_t_ha` (ou `_sc_ha`, `_kg_ha`)."""
    df = _read_table(data)
    pcol = _find(df, "produtividade", "prod", "rendimento")
    factor = SACA_KG / 1000 if "sc" in pcol or "saca" in pcol else (1 / 1000 if "kg" in pcol else 1.0)
    out = pd.DataFrame({
        "talhao": df[_find(df, "talhao", "talh", "area", "campo")].astype(str).str.strip(),
        "safra": df[_find(df, "safra", "ano", "year")].astype(str).str.strip(),
        "cultura_original": df[_find(df, "cultura", "crop")].astype(str).str.strip(),
        "produtividade_t_ha": _number(df[pcol]) * factor,
    })
    out["cultura"] = out["cultura_original"].map(crop_key)
    unknown = sorted(out.loc[out["cultura"].isna(), "cultura_original"].unique())
    out = out.dropna(subset=["cultura", "produtividade_t_ha"]).reset_index(drop=True)
    out.attrs["nao_reconhecidas"] = unknown
    return out


# ----------------------------------------------------------------------------- arquivo da fazenda

def dumps(state: dict) -> str:
    """Serializa os dados da fazenda. `talhoes` usa as chaves das culturas (ex.: "milho_2s")."""
    doc = {"formato": FORMATO, "salvo_em": dt.datetime.now().isoformat(timespec="seconds"), **state}
    return json.dumps(doc, ensure_ascii=False, indent=2, default=str)


def loads(text: str | bytes) -> dict:
    doc = json.loads(text)
    if not isinstance(doc, dict) or not str(doc.get("formato", "")).startswith("fazendareal/"):
        raise ValueError("este arquivo não é um arquivo de fazenda do RotaClima")
    prop = doc.get("propriedade") or {}
    if prop.get("lat") is not None and not (-60 <= float(prop["lat"]) <= 60 and -180 <= float(prop["lon"]) <= 180):
        raise ValueError("coordenadas fora do intervalo válido")
    for t in doc.get("talhoes", []):
        missing = [c for c in TALHAO_COLUMNS if c not in t]
        if missing:
            raise ValueError(f"talhão sem os campos {missing}")
        for c in ["verao_1", "segunda_1", "verao_2", "segunda_2"]:
            if t[c] not in CROPS:
                raise ValueError(f"cultura desconhecida: {t[c]}")
    return doc
