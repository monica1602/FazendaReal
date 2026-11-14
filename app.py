"""RotaClima — protótipo de apoio à decisão para rotação de culturas com dados da NASA.

Execute com:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
import re

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from rotaclima import diagnostics, engine, farmfile, modis, nasa_power, presets, recommender, theme
from rotaclima.crops import AWC_MM_PER_M, CROPS, SECOND, SUMMER, TEXTURES, crops_table
from rotaclima.engine import Scenario
from rotaclima.recommender import CRITERIA, PRESETS as WEIGHT_PRESETS, SLOT_LABELS, Economics, Talhao

st.set_page_config(page_title="RotaClima Fazenda Real — seus dados + dados da NASA", layout="wide",
                   initial_sidebar_state="expanded")

NAME_TO_KEY = {c.nome: k for k, c in CROPS.items()}
SUMMER_NAMES = [CROPS[k].nome for k in SUMMER]
SECOND_NAMES = [CROPS[k].nome for k in SECOND]
CROP_COLS = ["verao_1", "segunda_1", "verao_2", "segunda_2"]
TALHAO_COLS = presets.TALHAO_COLUMNS + ["argila_pct"]
RAIN_TEMPLATE = "data;chuva_mm\n01/10/2024;12,5\n02/10/2024;0\n03/10/2024;31\n"
YIELD_TEMPLATE = ("talhao;safra;cultura;produtividade_sc_ha\nTalhão 1;2022/23;Soja;62\n"
                  "Talhão 1;2023;Milho 2ª safra;95\nTalhão 2;2023/24;Soja;55\n")
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
PAGES = [
    "1. Propriedade", "2. Diagnóstico ambiental", "3. Mapa dos talhões", "4. Planejamento de rotação",
    "5. Comparação de cenários", "6. Simulador climático", "7. Relatório",
]
DISCLAIMER = ("Protótipo de apoio à decisão. As recomendações não substituem a avaliação de um engenheiro "
              "agrônomo, a análise de solo e o Zoneamento Agrícola de Risco Climático (ZARC).")

ss = st.session_state


# ----------------------------------------------------------------------------- estado

def brl(x: float) -> str:
    return ("R$ " + f"{x:,.0f}").replace(",", ".")


def esc(text: str) -> str:
    """Evita que "R$ ... R$" seja interpretado como fórmula LaTeX pelo Markdown do Streamlit."""
    return text.replace("$", "\\$")


def talhoes_frame(records: list[dict] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(records or [], columns=TALHAO_COLS)
    df = df.astype({"nome": "object", "lat": "float64", "lon": "float64", "area_ha": "float64", "textura": "object",
                    "ph": "float64", "mo_pct": "float64", "irrigado": "bool", "argila_pct": "float64",
                    **{c: "object" for c in CROP_COLS}})
    for c in CROP_COLS:
        df[c] = df[c].map(lambda k: CROPS[k].nome if k in CROPS else k)
    return df


def reference_eco() -> pd.DataFrame:
    return pd.DataFrame([{"key": k, "Cultura": c.nome, "Produtividade (t/ha)": c.prod_ref,
                          "Preço (R$/kg)": c.preco, "Custo (R$/ha)": c.custo} for k, c in CROPS.items()])


def new_farm() -> None:
    ss.preset = None
    ss.exemplo = False
    ss.prop_nome = ""
    ss.lat = ss.lon = None
    ss.talhoes = talhoes_frame()
    ss.talhoes_ver = ss.get("talhoes_ver", 0) + 1
    ss.polygons = {}
    ss.chuva_local = 0.0
    ss.rain_records = None
    ss.yields = None
    ss.summer_shift = 0
    ss.allowed = [k for k in CROPS if k != "pousio"]
    ss.weight_mode = list(WEIGHT_PRESETS)[2]
    ss.custom_weights = dict(WEIGHT_PRESETS[ss.weight_mode])
    ss.eco = reference_eco()
    ss.eco_ver = ss.get("eco_ver", 0) + 1
    ss.map_ver = ss.get("map_ver", 0) + 1
    ss.farm_file_ver = ss.get("farm_file_ver", 0) + 1


def load_preset(name: str) -> None:
    new_farm()
    p = presets.PRESETS[name]
    ss.preset = name
    ss.exemplo = True
    ss.lat, ss.lon = p["lat"], p["lon"]
    ss.prop_nome = f"Exemplo — {name.split(' — ')[0]}"
    ss.talhoes = talhoes_frame(presets.talhoes_records(name))


def farm_ready() -> bool:
    return ss.lat is not None and ss.lon is not None and not ss.talhoes.empty


def farm_state() -> dict:
    tal = ss.talhoes.copy()
    for c in CROP_COLS:
        tal[c] = tal[c].map(NAME_TO_KEY)
    rain = None
    if ss.rain_records is not None:
        rain = [{"data": f"{d:%Y-%m-%d}", "chuva_mm": float(v)}
                for d, v in zip(ss.rain_records["data"], ss.rain_records["chuva_mm"])]
    return {
        "exemplo": bool(ss.exemplo),
        "propriedade": {"nome": ss.prop_nome, "lat": ss.lat, "lon": ss.lon, "chuva_media_anual_mm": ss.chuva_local,
                        "deslocamento_semeadura_dias": int(ss.summer_shift)},
        "talhoes": json.loads(tal.to_json(orient="records")),
        "poligonos": {k: [list(p) for p in v] for k, v in ss.polygons.items()},
        "economia": [{"cultura": r["key"], "produtividade_t_ha": float(r["Produtividade (t/ha)"]),
                      "preco_r_kg": float(r["Preço (R$/kg)"]), "custo_r_ha": float(r["Custo (R$/ha)"])}
                     for _, r in ss.eco.iterrows()],
        "pluviometro": {"mensal": bool(ss.rain_records["mensal"].iloc[0]), "registros": rain} if rain else None,
        "produtividade": None if ss.yields is None else json.loads(ss.yields.to_json(orient="records")),
        "culturas_permitidas": list(ss.allowed),
        "prioridade": {"modo": ss.weight_mode, "pesos": dict(ss.custom_weights)},
    }


def apply_farm(doc: dict) -> None:
    new_farm()
    prop = doc.get("propriedade") or {}
    ss.exemplo = bool(doc.get("exemplo"))
    ss.prop_nome = str(prop.get("nome") or "")
    ss.lat = None if prop.get("lat") is None else float(prop["lat"])
    ss.lon = None if prop.get("lon") is None else float(prop["lon"])
    ss.chuva_local = float(prop.get("chuva_media_anual_mm") or 0.0)
    ss.summer_shift = int(prop.get("deslocamento_semeadura_dias") or 0)
    ss.talhoes = talhoes_frame(doc.get("talhoes"))
    ss.polygons = {k: [tuple(p) for p in v] for k, v in (doc.get("poligonos") or {}).items()}
    eco = {e["cultura"]: e for e in doc.get("economia") or [] if e.get("cultura") in CROPS}
    for i, r in ss.eco.iterrows():
        if r["key"] in eco:
            e = eco[r["key"]]
            ss.eco.loc[i, ["Produtividade (t/ha)", "Preço (R$/kg)", "Custo (R$/ha)"]] = [
                e["produtividade_t_ha"], e["preco_r_kg"], e["custo_r_ha"]]
    pluv = doc.get("pluviometro")
    if pluv and pluv.get("registros"):
        rec = pd.DataFrame(pluv["registros"])
        ss.rain_records = pd.DataFrame({"data": pd.to_datetime(rec["data"]), "chuva_mm": rec["chuva_mm"].astype(float),
                                        "mensal": bool(pluv.get("mensal"))})
    if doc.get("produtividade"):
        ss.yields = pd.DataFrame(doc["produtividade"])
    allowed = [k for k in doc.get("culturas_permitidas") or [] if k in CROPS]
    if allowed:
        ss.allowed = allowed
    prio = doc.get("prioridade") or {}
    if prio.get("modo") in list(WEIGHT_PRESETS) + ["Personalizado"]:
        ss.weight_mode = prio["modo"]
    if prio.get("pesos"):
        ss.custom_weights = {k: float(prio["pesos"].get(k, v)) for k, v in ss.custom_weights.items()}


def farm_name() -> str:
    return re.sub(r"[\\`*_#\[\]()<>|!~]", "", ss.prop_nome or "").strip()[:80] or "Minha fazenda"


def farm_filename() -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", farmfile._norm(ss.prop_nome or "fazenda")).strip("_") or "fazenda"
    return f"{slug}_{dt.date.today():%Y-%m-%d}.json"


def init_state() -> None:
    if "talhoes" not in ss:
        new_farm()
        ss.page = PAGES[0]


init_state()


# ----------------------------------------------------------------------------- dados

@st.cache_data(show_spinner="Consultando NASA POWER…", max_entries=16)
def load_climate(lat: float, lon: float) -> tuple[pd.DataFrame, dict]:
    return nasa_power.prepare(lat, lon)


def climate() -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    if ss.lat is None or ss.lon is None:
        return None, None
    try:
        return load_climate(round(ss.lat, 2), round(ss.lon, 2))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível obter dados da NASA POWER para este ponto ({exc}). "
                 "Verifique a conexão e tente novamente em alguns minutos.")
        return None, None


MIN_CALIB_MONTHS = 6


@st.cache_data(max_entries=8)
def rain_calibration_cached(lat: float, lon: float, records: pd.DataFrame) -> dict | None:
    df, _ = load_climate(lat, lon)
    return farmfile.rain_calibration(records, df["PRECTOTCORR"])


def rain_calibration() -> dict | None:
    if ss.rain_records is None or climate()[0] is None:
        return None
    return rain_calibration_cached(round(ss.lat, 2), round(ss.lon, 2), ss.rain_records)


def rain_factor() -> float:
    df, _ = climate()
    if df is None:
        return 1.0
    cal = rain_calibration()
    if cal and cal["meses"] >= MIN_CALIB_MONTHS:
        return cal["fator"]
    if not ss.chuva_local:
        return 1.0
    return float(ss.chuva_local) / diagnostics.annual_rain_mean(df)


def rain_source() -> str:
    cal = rain_calibration()
    if cal and cal["meses"] >= MIN_CALIB_MONTHS:
        return f"{cal['meses']} meses de registros do pluviômetro da fazenda (fator × {cal['fator']:.2f})"
    if ss.chuva_local:
        return f"média anual informada pelo produtor, {ss.chuva_local:.0f} mm (fator × {rain_factor():.2f})"
    return ""


def climate_adjusted() -> pd.DataFrame | None:
    df, meta = climate()
    if df is None:
        return None
    f = rain_factor()
    return engine.apply_scenario(df, Scenario(), ss.lat, meta["elevacao_m"], f) if f != 1.0 else df


def talhao_obj(row: pd.Series) -> Talhao:
    rot = tuple(NAME_TO_KEY[row[c]] for c in ["verao_1", "segunda_1", "verao_2", "segunda_2"])
    return Talhao(str(row["nome"]), float(row["area_ha"]), str(row["textura"]), float(row["ph"]),
                  float(row["mo_pct"]), bool(row["irrigado"]), rot, float(row["lat"]), float(row["lon"]))


def talhao_tuple(t: Talhao) -> tuple:
    return (t.nome, t.area_ha, t.textura, t.ph, t.mo_pct, t.irrigado, t.rotacao_atual, t.lat, t.lon)


def eco_tuple() -> tuple:
    return tuple((r["key"], float(r["Produtividade (t/ha)"]), float(r["Preço (R$/kg)"]), float(r["Custo (R$/ha)"]))
                 for _, r in ss.eco.iterrows())


def economics_from(eco_t: tuple) -> Economics:
    return Economics(prod_ref={k: p for k, p, _, _ in eco_t}, preco={k: v for k, _, v, _ in eco_t},
                     custo={k: c for k, _, _, c in eco_t})


@st.cache_data(show_spinner="Simulando balanço hídrico e avaliando rotações…", max_entries=64)
def evaluate_cached(lat: float, lon: float, t_tuple: tuple, allowed: tuple, eco_t: tuple, sc: Scenario,
                    rfactor: float, summer_shift: int, only: tuple | None = None) -> pd.DataFrame:
    df, meta = load_climate(lat, lon)
    clim = engine.apply_scenario(df, sc, lat, meta["elevacao_m"], rfactor)
    t = Talhao(*t_tuple)
    runner = engine.CropRunner(clim, lat, t.awc, t.irrigado, sc.limite_irrigacao_mm, summer_shift)
    include = [t.rotacao_atual] + list(only or [])
    return recommender.evaluate(t, runner, economics_from(eco_t), set(allowed), summer_shift,
                                include=include, only_include=only is not None)


def evaluate_talhao(t: Talhao, sc: Scenario | None = None, only: tuple | None = None) -> pd.DataFrame:
    return evaluate_cached(round(ss.lat, 2), round(ss.lon, 2), talhao_tuple(t), tuple(sorted(ss.allowed)),
                           eco_tuple(), sc or Scenario(), rain_factor(), int(ss.summer_shift), only)


def current_weights() -> dict[str, float]:
    if ss.weight_mode in WEIGHT_PRESETS:
        return WEIGHT_PRESETS[ss.weight_mode]
    return ss.custom_weights


def select_talhao(label: str = "Talhão") -> Talhao | None:
    if ss.talhoes.empty:
        st.warning("Cadastre ao menos um talhão na tela 1.")
        return None
    names = ss.talhoes["nome"].astype(str).tolist()
    cur = ss.get("talhao_atual", names[0])
    idx = names.index(cur) if cur in names else 0
    sel = st.selectbox(label, names, index=idx)
    ss.talhao_atual = sel
    return talhao_obj(ss.talhoes[ss.talhoes["nome"].astype(str) == sel].iloc[0])


def rotation_timeline(rot: tuple[str, ...], delays: tuple[int, int], title: str = "") -> go.Figure:
    year = dt.date.today().year
    rows = []
    for i, key in enumerate(rot):
        crop = CROPS[key]
        y = year if i < 2 else year + 1
        delay = delays[0] if i == 1 else delays[1] if i == 3 else 0
        start = engine.sow_date(crop, y, ss.lat, int(ss.summer_shift)) + pd.Timedelta(days=delay)
        rows.append({"Época": SLOT_LABELS[i], "Cultura": crop.nome, "Início": start,
                     "Fim": start + pd.Timedelta(days=crop.ciclo), "Família": crop.familia})
    df = pd.DataFrame(rows)
    fig = px.timeline(df, x_start="Início", x_end="Fim", y="Época", color="Cultura", text="Cultura",
                      color_discrete_sequence=PALETTE, hover_data=["Família"])
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30 if title else 10, b=10), showlegend=False,
                      title=title or None)
    return fig


def criteria_bar(row: pd.Series) -> go.Figure:
    vals = [row[k] for k in CRITERIA]
    fig = go.Figure(go.Bar(x=vals, y=list(CRITERIA.values()), orientation="h", marker_color=PALETTE[:5],
                           text=[f"{v:.0f}" for v in vals], textposition="outside"))
    fig.update_layout(height=220, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 110], title="0–100"))
    return fig


def base_map(center: tuple[float, float], zoom: float, satellite: bool) -> go.Figure:
    fig = go.Figure()
    layout = dict(center=dict(lat=center[0], lon=center[1]), zoom=zoom)
    if satellite:
        layout["style"] = "white-bg"
        layout["layers"] = [{"below": "traces", "sourcetype": "raster", "source": [ESRI_TILES],
                             "sourceattribution": "Esri, Maxar, Earthstar Geographics"}]
    else:
        layout["style"] = "open-street-map"
    fig.update_layout(map=layout, margin=dict(l=0, r=0, t=0, b=0), height=480, showlegend=True,
                      legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    return fig


def talhao_polygon(row: pd.Series) -> list[tuple[float, float]]:
    poly = ss.polygons.get(str(row["nome"]))
    return poly or presets.square_polygon(float(row["lat"]), float(row["lon"]), float(row["area_ha"]))


def add_talhao_layer(fig: go.Figure, values: dict[str, float | str], colors: dict[str, str], fmt) -> None:
    for _, row in ss.talhoes.iterrows():
        name = str(row["nome"])
        poly = talhao_polygon(row)
        val = values.get(name)
        fig.add_trace(go.Scattermap(
            lat=[p[0] for p in poly], lon=[p[1] for p in poly], mode="lines", fill="toself",
            fillcolor=colors.get(name, "rgba(200,200,200,0.4)"), line=dict(color="white", width=2),
            name=f"{name}: {fmt(val)}", hoverinfo="name"))
        fig.add_trace(go.Scattermap(lat=[row["lat"]], lon=[row["lon"]], mode="text", text=[name.split(" — ")[0]],
                                    textfont=dict(color="white", size=13), showlegend=False, hoverinfo="skip"))


def color_scale(values: dict[str, float], scale: str = "RdYlGn", reverse: bool = False,
                vmin: float | None = None, vmax: float | None = None) -> dict[str, str]:
    nums = [v for v in values.values() if v is not None and not np.isnan(v)]
    if not nums:
        return {}
    lo = vmin if vmin is not None else min(nums)
    hi = vmax if vmax is not None else max(nums)
    out = {}
    for k, v in values.items():
        if v is None or np.isnan(v):
            continue
        x = 0.5 if hi == lo else (v - lo) / (hi - lo)
        x = 1 - x if reverse else x
        c = px.colors.sample_colorscale(scale, [min(max(x, 0), 1)])[0]
        out[k] = c.replace("rgb", "rgba").replace(")", ",0.65)")
    return out


# ----------------------------------------------------------------------------- barra lateral

PALETTE = list(theme.THEMES[ss.page].paleta)
st.markdown(theme.css(ss.page), unsafe_allow_html=True)


def hero(title: str, text: str | None = None) -> None:
    t = theme.THEMES[ss.page]
    with st.container(key="hero"):
        c1, c2 = st.columns([2.4, 1], vertical_alignment="center")
        with c1:
            st.markdown(f'<span class="rc-chip">{t.icone} {esc(t.resumo)}</span>', unsafe_allow_html=True)
            st.title(title)
            if text:
                st.markdown(text)
        c2.markdown(theme.hero_image(ss.page), unsafe_allow_html=True)


with st.sidebar:
    st.markdown(theme.brand_html(), unsafe_allow_html=True)
    st.caption("Rotação de culturas orientada por dados da NASA")
    st.radio("Navegação", PAGES, key="page", label_visibility="collapsed", format_func=theme.nav_label)
    st.divider()
    if ss.lat is None and ss.talhoes.empty:
        st.markdown("**Nenhuma fazenda cadastrada**")
        st.caption("Comece na tela 1: informe a localização e os talhões.")
    else:
        st.markdown(f"**{farm_name()}**")
        loc = f"lat {ss.lat:.2f}, lon {ss.lon:.2f}" if ss.lat is not None else "sem localização"
        st.caption(f"{len(ss.talhoes)} talhões · {ss.talhoes['area_ha'].sum():.0f} ha · {loc}")
        if ss.exemplo:
            st.warning("Exemplo com talhões, solo e preços fictícios.")
        _df, _meta = climate()
        if _meta:
            st.caption(f"Clima: NASA POWER ({_meta['inicio']:%Y}–{_meta['fim']:%d/%m/%Y}) · {_meta['origem']}")
        st.download_button("💾 Salvar arquivo da fazenda", farmfile.dumps(farm_state()).encode("utf-8"),
                           farm_filename(), "application/json", width="stretch", key="save_sidebar",
                           help="Baixa um arquivo com todos os dados que você informou. Na próxima visita, "
                                "abra esse arquivo na tela 1 para continuar de onde parou.")
    st.divider()
    st.caption(DISCLAIMER)


# ----------------------------------------------------------------------------- 1. propriedade

def start_panel() -> None:
    with st.container(border=True):
        st.markdown("**Como você quer começar?**")
        a, b, c = st.columns(3)
        with a:
            up = st.file_uploader("📂 Abrir o arquivo da fazenda (.json)", type=["json"],
                                  key=f"farm_file_{ss.get('farm_file_ver', 0)}",
                                  help="Arquivo salvo pelo botão 💾 em uma visita anterior.")
            if up is not None:
                backup = farm_state()
                try:
                    apply_farm(farmfile.loads(up.getvalue()))
                except (ValueError, KeyError, TypeError) as exc:
                    apply_farm(backup)
                    st.error(f"Não foi possível abrir o arquivo: {exc}")
                else:
                    st.rerun()
        with b:
            st.markdown("🆕 **Cadastrar uma fazenda nova**")
            st.caption("Começa com tudo em branco. Os dados ficam só no seu navegador enquanto o site estiver aberto.")
            if st.button("Começar do zero", width="stretch"):
                new_farm()
                st.rerun()
        with c:
            st.markdown("👀 **Só conhecer o site**")
            ex = st.selectbox("Exemplo", list(presets.PRESETS), label_visibility="collapsed")
            if st.button("Carregar exemplo", width="stretch"):
                load_preset(ex)
                st.rerun()


def add_polygons(polys: list[tuple[str, list[tuple[float, float]]]]) -> None:
    rows = []
    for name, poly in polys:
        lat, lon, area = farmfile.polygon_stats(poly)
        ss.polygons[name] = poly
        rows.append({"nome": name, "lat": round(lat, 5), "lon": round(lon, 5), "area_ha": round(area, 1),
                     "textura": "média", "ph": 5.5, "mo_pct": 2.0, "irrigado": False, "argila_pct": np.nan,
                     "verao_1": "soja", "segunda_1": "milho_2s", "verao_2": "soja", "segunda_2": "milho_2s"})
    new = talhoes_frame(rows).set_index("nome")
    tal = ss.talhoes.set_index("nome")
    existing = new.index.intersection(tal.index)
    tal.loc[existing, ["lat", "lon", "area_ha"]] = new.loc[existing, ["lat", "lon", "area_ha"]]
    added = new.drop(existing)
    ss.talhoes = (pd.concat([tal, added]) if not tal.empty else added).reset_index()
    if ss.lat is None:
        ss.lat, ss.lon = round(float(new["lat"].mean()), 4), round(float(new["lon"].mean()), 4)
    ss.talhoes_ver += 1
    ss.map_ver += 1


def next_talhao_names(n: int) -> list[str]:
    used, out, i = set(ss.talhoes["nome"].astype(str)), [], 1
    while len(out) < n:
        if f"Talhão {i}" not in used:
            out.append(f"Talhão {i}")
        i += 1
    return out


def farm_map() -> None:
    drawing = st.radio("No mapa:", ["📍 Clicar para marcar a sede", "✏️ Desenhar os talhões"], horizontal=True,
                       key="map_mode", label_visibility="collapsed") == "✏️ Desenhar os talhões"
    has_loc = ss.lat is not None
    center = [ss.lat, ss.lon] if has_loc else [-14.5, -52.0]
    m = folium.Map(location=center, zoom_start=15 if has_loc else 4, tiles=None, control_scale=True)
    folium.TileLayer(ESRI_TILES, attr="Esri World Imagery", name="Satélite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Mapa com estradas e cidades", show=False).add_to(m)
    if has_loc:
        folium.Marker(center, tooltip="Sede: ponto usado para buscar o clima na NASA POWER",
                      icon=folium.Icon(color="green", icon="home")).add_to(m)
    for _, r in ss.talhoes.dropna(subset=["lat", "lon", "area_ha"]).iterrows():
        folium.Polygon(talhao_polygon(r), color="#FFD54F", weight=2, fill=True, fill_opacity=0.15,
                       tooltip=str(r["nome"])).add_to(m)
    if drawing:
        Draw(draw_options={"polygon": True, "rectangle": True, "polyline": False, "circle": False, "marker": False,
                           "circlemarker": False}, edit_options={"edit": False, "remove": True}).add_to(m)
    folium.LayerControl().add_to(m)
    out = st_folium(m, key=f"farm_map_{ss.map_ver}_{drawing}", height=430, use_container_width=True,
                    returned_objects=["last_clicked", "all_drawings"]) or {}
    if drawing:
        st.caption("Use o ícone de polígono ou retângulo à esquerda do mapa e clique nos cantos do talhão; "
                   "clique no primeiro ponto para fechar.")
        feats = [f for f in out.get("all_drawings") or [] if (f.get("geometry") or {}).get("type") == "Polygon"]
        if feats and st.button(f"Adicionar {len(feats)} talhão(ões) desenhado(s)", type="primary"):
            names = next_talhao_names(len(feats))
            add_polygons([(n, poly) for n, (_, poly) in zip(names, farmfile.parse_geojson(
                {"type": "FeatureCollection", "features": feats}))])
            st.rerun()
    else:
        st.caption("Dê zoom até a sua fazenda e clique na sede. A camada de satélite ajuda a encontrar os talhões.")
        lc = out.get("last_clicked")
        if lc and (lc["lat"], lc["lng"]) != ss.get("last_click"):
            ss.last_click = (lc["lat"], lc["lng"])
            ss.lat, ss.lon = round(lc["lat"], 4), round(lc["lng"], 4)
            st.rerun()


def rain_panel(df: pd.DataFrame) -> None:
    st.markdown("**🌧️ Pluviômetro da fazenda (opcional)**")
    st.caption("A chuva da NASA POWER é a média de uma área de ~55 km. Com os registros do seu pluviômetro, "
               "o app ajusta a chuva para a sua fazenda.")
    up = st.file_uploader("Planilha CSV com as colunas `data` e `chuva_mm` (diária ou mensal)", type=["csv", "txt"],
                          key="rain_file")
    st.download_button("Baixar modelo de planilha", RAIN_TEMPLATE.encode("utf-8"), "modelo_pluviometro.csv",
                       key="rain_tpl")
    if up is not None and up.file_id != ss.get("rain_file_id"):
        ss.rain_file_id = up.file_id
        try:
            ss.rain_records = farmfile.read_rain_csv(up.getvalue())
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(f"Não foi possível ler a planilha: {exc}")
    if ss.rain_records is not None:
        cal = rain_calibration()
        r = ss.rain_records
        st.write(f"{len(r)} registros de {r['data'].min():%m/%Y} a {r['data'].max():%m/%Y}.")
        if cal is None:
            st.warning("Os registros não coincidem com meses completos da série da NASA POWER.")
        elif cal["meses"] < MIN_CALIB_MONTHS:
            st.warning(f"Só {cal['meses']} meses completos em comum; são necessários {MIN_CALIB_MONTHS} para ajustar.")
        else:
            st.success(f"Ajuste aplicado: chuva da NASA × {cal['fator']:.2f} ({cal['meses']} meses comparados).")
            t = cal["tabela"]
            fig = go.Figure()
            fig.add_bar(x=t.index, y=t["pluviometro"], name="Pluviômetro", marker_color="#2E7D32")
            fig.add_bar(x=t.index, y=t["nasa_power"], name="NASA POWER", marker_color="#4C8BF5")
            fig.update_layout(height=230, barmode="group", margin=dict(t=10, b=10, l=10, r=10), yaxis_title="mm/mês",
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
        if st.button("Remover registros do pluviômetro"):
            ss.rain_records = None
            st.rerun()
    elif ss.lat is not None:
        power_mean = diagnostics.annual_rain_mean(df)
        ss.chuva_local = st.number_input(
            "Sem planilha? Informe a chuva média anual da fazenda (mm)", 0.0, 5000.0, float(ss.chuva_local), 50.0,
            help=f"A NASA POWER estima {power_mean:.0f} mm/ano para esta área (0 = sem ajuste).")


def yields_panel() -> None:
    st.caption("Informe o que foi colhido em cada talhão. O app mostra o histórico e pode usar a média da fazenda "
               "como produtividade de referência nos cálculos de margem.")
    up = st.file_uploader("Planilha CSV: `talhao`, `safra`, `cultura` e `produtividade_sc_ha` "
                          "(ou `produtividade_t_ha` / `produtividade_kg_ha`)", type=["csv", "txt"], key="yield_file")
    st.download_button("Baixar modelo de planilha", YIELD_TEMPLATE.encode("utf-8"), "modelo_produtividade.csv",
                       key="yield_tpl")
    if up is not None and up.file_id != ss.get("yield_file_id"):
        ss.yield_file_id = up.file_id
        try:
            ss.yields = farmfile.read_yield_csv(up.getvalue())
            if ss.yields.attrs.get("nao_reconhecidas"):
                st.warning("Culturas não reconhecidas (ignoradas): " + ", ".join(ss.yields.attrs["nao_reconhecidas"]))
        except (ValueError, KeyError) as exc:
            st.error(f"Não foi possível ler a planilha: {exc}")
    if ss.yields is None or ss.yields.empty:
        return
    y = ss.yields.assign(Cultura=ss.yields["cultura"].map(lambda k: CROPS[k].nome))
    fig = px.bar(y, x="safra", y="produtividade_t_ha", color="talhao", facet_col="Cultura", barmode="group",
                 labels={"produtividade_t_ha": "t/ha", "safra": "Safra", "talhao": "Talhão"},
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    means = y.groupby("cultura")["produtividade_t_ha"].mean()
    st.dataframe(pd.DataFrame({"Cultura": [CROPS[k].nome for k in means.index],
                               "Média colhida (t/ha)": means.round(2).values,
                               "Referência atual (t/ha)": [float(ss.eco.loc[ss.eco["key"] == k, "Produtividade (t/ha)"]
                                                                 .iloc[0]) for k in means.index]}),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    if c1.button("Usar a média colhida como produtividade de referência", type="primary"):
        for k, v in means.items():
            ss.eco.loc[ss.eco["key"] == k, "Produtividade (t/ha)"] = round(float(v), 2)
        ss.eco_ver += 1
        st.rerun()
    if c2.button("Remover produtividades"):
        ss.yields = None
        st.rerun()


def page_propriedade() -> None:
    hero("1. Cadastro da propriedade",
         "Informe os dados reais da sua fazenda: localização, talhões, análise de solo, chuva e produtividade. "
         "O clima vem automaticamente da **NASA POWER** e a vegetação do satélite **MODIS**.")
    start_panel()
    if ss.exemplo:
        st.warning("Você está vendo um **exemplo**: talhões, solo e preços são fictícios (o clima e o NDVI são reais). "
                   "Para usar a sua fazenda, clique em **Começar do zero**.")

    st.subheader("📍 Localização")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        ss.prop_nome = st.text_input("Nome da propriedade", ss.prop_nome, placeholder="Ex.: Sítio Boa Esperança")
        a, b = st.columns(2)
        lat = a.number_input("Latitude", -60.0, 60.0, ss.lat, 0.01, format="%.4f", placeholder="Ex.: -17.7800",
                             help="Graus decimais; no Sul do Equador o valor é negativo.")
        lon = b.number_input("Longitude", -180.0, 180.0, ss.lon, 0.01, format="%.4f", placeholder="Ex.: -51.0500",
                             help="Graus decimais; no Brasil o valor é negativo.")
        if (lat, lon) != (ss.lat, ss.lon) and None not in (lat, lon):
            ss.lat, ss.lon = lat, lon
            ss.map_ver += 1
            st.rerun()
        farm_map()
    with c2:
        df, meta = climate()
        with st.container(border=True):
            st.markdown("**🛰️ Clima da NASA POWER**")
            if meta:
                st.write(f"Período: {meta['inicio']:%d/%m/%Y} a {meta['fim']:%d/%m/%Y} · origem: {meta['origem']}")
                st.write(f"Célula MERRA-2 ~0,5° × 0,625° (≈ 55 × 65 km) · elevação média {meta['elevacao_m']:.0f} m")
            elif ss.lat is None:
                st.caption("Informe a localização para buscar 25 anos de clima diário da NASA (leva ~10–30 s).")
        if meta:
            with st.container(border=True):
                rain_panel(df)
        shift = st.slider("Data de semeadura das culturas de verão", -20, 30, int(ss.summer_shift), 5,
                          format="%+d dias", help="Deslocamento em relação a 15/out (referência).")
        ss.summer_shift = shift
        st.caption(f"Semeadura de verão: {(dt.date(2000, 10, 15) + dt.timedelta(days=shift)):%d/%m}")

    st.subheader("🌱 Talhões e análise de solo")
    st.caption("Adicione uma linha por talhão (botão **+** no fim da tabela) ou desenhe os talhões no mapa acima. "
               "Use os dados da **análise de solo**: se preencher a argila, a textura é definida automaticamente. "
               "As colunas de cultura são a **rotação atual** (2 anos), usada como comparação.")
    edited = st.data_editor(
        ss.talhoes, key=f"ed_talhoes_{ss.talhoes_ver}", num_rows="dynamic", width="stretch", hide_index=True,
        column_order=["nome", "area_ha", "argila_pct", "textura", "ph", "mo_pct", "irrigado", *CROP_COLS, "lat", "lon"],
        column_config={
            "nome": st.column_config.TextColumn("Talhão", required=True),
            "lat": st.column_config.NumberColumn("Lat.", format="%.4f", help="Vazio = localização da sede"),
            "lon": st.column_config.NumberColumn("Lon.", format="%.4f", help="Vazio = localização da sede"),
            "area_ha": st.column_config.NumberColumn("Área (ha)", min_value=0.1, required=True),
            "argila_pct": st.column_config.NumberColumn("Argila (%)", min_value=0.0, max_value=100.0, step=1.0,
                                                        help="Da análise granulométrica (opcional)."),
            "textura": st.column_config.SelectboxColumn("Textura", options=TEXTURES),
            "ph": st.column_config.NumberColumn("pH (H₂O)", min_value=3.5, max_value=9.0, step=0.1),
            "mo_pct": st.column_config.NumberColumn("MO (%)", min_value=0.0, max_value=15.0, step=0.1),
            "irrigado": st.column_config.CheckboxColumn("Irrigado"),
            "verao_1": st.column_config.SelectboxColumn("Verão ano 1", options=SUMMER_NAMES),
            "segunda_1": st.column_config.SelectboxColumn("2ª safra ano 1", options=SECOND_NAMES),
            "verao_2": st.column_config.SelectboxColumn("Verão ano 2", options=SUMMER_NAMES),
            "segunda_2": st.column_config.SelectboxColumn("2ª safra ano 2", options=SECOND_NAMES),
        })
    edited = edited.dropna(subset=["nome", "area_ha"]).reset_index(drop=True)
    defaults = {"lat": ss.lat, "lon": ss.lon, "textura": "média", "ph": 5.5, "mo_pct": 2.0, "irrigado": False,
                "verao_1": CROPS["soja"].nome, "segunda_1": CROPS["milho_2s"].nome, "verao_2": CROPS["soja"].nome,
                "segunda_2": CROPS["milho_2s"].nome}
    edited = edited.fillna({k: v for k, v in defaults.items() if v is not None})
    has_clay = edited["argila_pct"].notna()
    edited.loc[has_clay, "textura"] = edited.loc[has_clay, "argila_pct"].map(farmfile.textura_from_argila)
    edited = edited.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if not edited.equals(ss.talhoes.reset_index(drop=True)):
        ss.talhoes = edited
        ss.talhoes_ver += 1
        st.rerun()
    if ss.talhoes.empty:
        st.info("Nenhum talhão ainda. " + ("Informe a localização primeiro." if ss.lat is None else
                                           "Clique no **+** da tabela ou desenhe no mapa."))

    with st.expander("📐 Importar limites dos talhões (GeoJSON ou KML do Google Earth)"):
        st.caption("Polígonos com nome. Talhões novos recebem valores de solo padrão para você editar na tabela.")
        up = st.file_uploader("Arquivo GeoJSON ou KML", type=["geojson", "json", "kml"], key="limits_file")
        if up is not None and st.button("Importar"):
            try:
                data = up.getvalue()
                polys = (farmfile.parse_kml(data) if up.name.lower().endswith(".kml")
                         else farmfile.parse_geojson(json.loads(data)))
                add_polygons(polys)
                st.success(f"{len(polys)} talhões importados.")
                st.rerun()
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Arquivo inválido: {exc}")

    with st.expander("🌾 Produtividade colhida nas últimas safras (opcional)"):
        yields_panel()

    with st.expander("💰 Preços, custos e produtividade de referência — troque pelos valores da sua fazenda"):
        eco = st.data_editor(ss.eco, key=f"ed_eco_{ss.eco_ver}", hide_index=True, width="stretch",
                             disabled=["key", "Cultura"], column_config={"key": None})
        if not eco.equals(ss.eco):
            ss.eco = eco
            ss.eco_ver += 1
            st.rerun()
        changed = (eco.drop(columns="Cultura").set_index("key") != reference_eco().drop(columns="Cultura")
                   .set_index("key")).any(axis=1)
        st.caption(f"{int(changed.sum())} de {len(changed)} culturas com valores alterados. "
                   "Margem = produtividade estimada × preço − custo − irrigação (R$ 1,50/mm/ha). "
                   "Coberturas não geram receita direta.")
    with st.expander("Base agronômica utilizada pelo motor"):
        st.dataframe(crops_table().drop(columns=["key"]), hide_index=True, width="stretch")
        st.caption("Kc: FAO-56 (Allen et al., 1998). Ky: FAO-33 (Doorenbos & Kassam, 1979). Água disponível por "
                   f"textura (mm/m): {AWC_MM_PER_M}. Limiares de calor/geada e fatores de perda são heurísticos.")

    with st.container(border=True):
        st.markdown("**💾 Guardar os dados**")
        st.caption("Por privacidade, o site **não guarda** os dados da sua fazenda: eles ficam só no seu navegador "
                   "e somem ao fechar a página. Baixe o arquivo da fazenda e abra-o aqui na próxima visita. "
                   "As coordenadas são enviadas à NASA e ao ORNL DAAC apenas para buscar clima e NDVI.")
        st.download_button("💾 Baixar arquivo da fazenda", farmfile.dumps(farm_state()).encode("utf-8"),
                           farm_filename(), "application/json", type="primary", key="save_page")


# ----------------------------------------------------------------------------- 2. diagnóstico

def page_diagnostico() -> None:
    hero("2. Diagnóstico ambiental", "Como a água e o calor se comportaram nos últimos 25 anos no ponto da "
         "propriedade, com dados diários da **NASA POWER**.")
    df = climate_adjusted()
    if df is None:
        return
    _, meta = climate()
    clim = diagnostics.monthly_climatology(df)
    annual = diagnostics.annual_series(df)
    spells = diagnostics.wet_season_dry_spells(df, ss.lat)

    k = st.columns(6)
    k[0].metric("🌧️ Chuva anual média", f"{annual['chuva'].mean():.0f} mm")
    k[1].metric("💨 ET0 anual (FAO-56)", f"{annual['et0'].mean():.0f} mm", help="Evapotranspiração de referência "
                "calculada por Penman-Monteith com temperatura, umidade, radiação e vento da NASA POWER.")
    k[2].metric("⚖️ Balanço chuva − ET0", f"{annual['balanco'].mean():+.0f} mm")
    k[3].metric("🌡️ Temperatura média", f"{annual['temp_media'].mean():.1f} °C")
    k[4].metric("🔥 Dias/ano > 35 °C", f"{annual['dias_calor'].mean():.0f}")
    k[5].metric("☀️ Maior veranico (dez–fev)", f"{spells.mean():.0f} dias", help="Média do maior período "
                "consecutivo com chuva < 1 mm entre dezembro e fevereiro. Dados em grade (≈ 55 km) "
                "suavizam a chuva e tendem a subestimar veranicos locais; confira com pluviômetro.")

    with st.container(border=True):
        st.markdown("**O que os dados indicam**")
        for line in diagnostics.summary_text(df, ss.lat):
            st.markdown(f"- {line}")
        if rain_source():
            st.caption(f"Chuva calibrada com {rain_source()}.")

    c1, c2 = st.columns(2)
    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=clim.index, y=clim["PRECTOTCORR"], name="Chuva (mm/mês)", marker_color="#4C8BF5",
                    error_y=dict(type="data", symmetric=False, array=clim["P90_chuva"] - clim["PRECTOTCORR"],
                                 arrayminus=clim["PRECTOTCORR"] - clim["P10_chuva"], color="#9ab"))
        fig.add_scatter(x=clim.index, y=clim["ET0"], name="ET0 (mm/mês)", line=dict(color="#E8710A", width=3))
        fig.add_scatter(x=clim.index, y=clim["T2M_MAX"], name="T máx (°C)", line=dict(color="#C62828", dash="dot"),
                        secondary_y=True)
        fig.add_scatter(x=clim.index, y=clim["T2M_MIN"], name="T mín (°C)", line=dict(color="#1565C0", dash="dot"),
                        secondary_y=True)
        fig.update_layout(title="Climatologia mensal (barras: média e faixa P10–P90)", height=380,
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
        fig.update_yaxes(title_text="mm/mês", secondary_y=False)
        fig.update_yaxes(title_text="°C", secondary_y=True)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=annual.index, y=annual["chuva"], name="Chuva anual", marker_color="#4C8BF5")
        fig.add_hline(y=annual["chuva"].mean(), line_dash="dash", annotation_text="média")
        fig.add_scatter(x=annual.index, y=annual["et0"], name="ET0 anual", line=dict(color="#E8710A"))
        fig.update_layout(title="Chuva e ET0 por ano", height=380, legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=50, b=10), yaxis_title="mm")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        an = diagnostics.recent_anomaly(df)
        fig = go.Figure(go.Bar(x=an.index, y=an["anomalia"],
                               marker_color=np.where(an["anomalia"] >= 0, "#2E7D32", "#C62828"),
                               customdata=np.c_[an["chuva"], an["normal"]],
                               hovertemplate="%{x|%b/%Y}<br>chuva %{customdata[0]:.0f} mm<br>normal "
                                             "%{customdata[1]:.0f} mm<extra></extra>"))
        fig.update_layout(title="Anomalia de chuva — últimos 24 meses (mm)", height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with c4:
        g = df["GWETROOT"].dropna()
        doy_mean = g.groupby(g.index.dayofyear).mean()
        last_year = g[g.index.year == g.index.max().year]
        fig = go.Figure()
        fig.add_scatter(x=doy_mean.index, y=doy_mean.values, name="média histórica", line=dict(color="#777"))
        fig.add_scatter(x=last_year.index.dayofyear, y=last_year.values, name=str(g.index.max().year),
                        line=dict(color="#2E7D32", width=3))
        fig.update_layout(title="Umidade do solo na zona radicular (GWETROOT, 0–1)", height=330,
                          xaxis_title="dia do ano", margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Fonte e limitações dos dados"):
        st.markdown(f"""
- **NASA POWER** (Prediction Of Worldwide Energy Resources), API diária, comunidade *AG*, ponto
  ({meta['lat']}, {meta['lon']}). Parâmetros: {', '.join(nasa_power.PARAMETERS)}.
  Documentação: <https://power.larc.nasa.gov/docs/>.
- Resolução espacial grosseira (MERRA-2 ~0,5° × 0,625°; radiação CERES 1°): todos os talhões compartilham o
  mesmo clima. Diferenças entre talhões vêm do solo informado e do NDVI MODIS.
- Precipitação de reanálise pode subestimar ou superestimar a chuva local — use a calibração com pluviômetro.
- GWETROOT é um índice de saturação do modelo MERRA-2, não uma medição de campo. Para umidade do solo por
  satélite, a evolução natural é integrar o SMAP (resolução de ~9–36 km).
""")


# ----------------------------------------------------------------------------- 3. mapa

def soil_vulnerability(row: pd.Series, ndvi_mean: float | None) -> tuple[int, str]:
    score = {"arenosa": 2, "média": 1}.get(row["textura"], 0)
    score += 2 if row["mo_pct"] < 1.5 else 1 if row["mo_pct"] < 2.5 else 0
    score += 1 if row["ph"] < 5.0 else 0
    if ndvi_mean is not None and not np.isnan(ndvi_mean) and ndvi_mean < 0.45:
        score += 1
    return score, "alta" if score >= 4 else "média" if score >= 2 else "baixa"


def page_mapa() -> None:
    hero("3. Mapa dos talhões",
         "Visualização dos talhões com indicadores de solo (informados) e de vegetação por satélite "
         "(**NDVI MODIS MOD13Q1**, 250 m, composição de 16 dias).")
    ndvi = {str(r["nome"]): modis.cached_ndvi(float(r["lat"]), float(r["lon"])) for _, r in ss.talhoes.iterrows()}
    ndvi_last = {k: (float(v["ndvi"].dropna().iloc[-1]) if v is not None and v["ndvi"].notna().any() else np.nan)
                 for k, v in ndvi.items()}
    ndvi_mean = {k: (float(v["ndvi"].mean()) if v is not None else np.nan) for k, v in ndvi.items()}

    c1, c2 = st.columns([3, 1])
    with c2:
        indicador = st.radio("Indicador", ["NDVI mais recente", "NDVI médio (12 meses)", "Vulnerabilidade do solo",
                                           "Matéria orgânica (%)", "pH", "Água disponível no solo (mm)"])
        satellite = st.toggle("Imagem de satélite", value=True)
        missing = [k for k, v in ndvi.items() if v is None]
        if missing:
            st.info(f"NDVI ainda não baixado para: {', '.join(missing)}")
        if st.button("Baixar/atualizar NDVI MODIS", help="ORNL DAAC MODIS Web Service; ~10–60 s por talhão."):
            prog = st.progress(0.0)
            for i, (_, r) in enumerate(ss.talhoes.iterrows()):
                try:
                    modis.fetch_ndvi_point(float(r["lat"]), float(r["lon"]), use_cache=False)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"{r['nome']}: falha ao consultar MODIS ({exc})")
                prog.progress((i + 1) / len(ss.talhoes))
            st.rerun()

    vul = {}
    for _, r in ss.talhoes.iterrows():
        vul[str(r["nome"])] = soil_vulnerability(r, ndvi_mean.get(str(r["nome"])))
    if indicador.startswith("NDVI"):
        vals = ndvi_last if "recente" in indicador else ndvi_mean
        colors = color_scale(vals, "RdYlGn", vmin=0.1, vmax=0.9)
        fmt = lambda v: "sem dado" if v is None or np.isnan(v) else f"{v:.2f}"  # noqa: E731
    elif indicador.startswith("Vulnerabilidade"):
        vals = {k: float(v[0]) for k, v in vul.items()}
        colors = color_scale(vals, "RdYlGn", reverse=True, vmin=0, vmax=6)
        fmt = lambda v: f"{v:.0f}/6"  # noqa: E731
    else:
        col = {"Matéria orgânica (%)": "mo_pct", "pH": "ph"}.get(indicador)
        if col:
            vals = {str(r["nome"]): float(r[col]) for _, r in ss.talhoes.iterrows()}
        else:
            vals = {str(r["nome"]): AWC_MM_PER_M[r["textura"]] * 1.0 for _, r in ss.talhoes.iterrows()}
        colors = color_scale(vals, "RdYlGn")
        fmt = lambda v: f"{v:.1f}"  # noqa: E731
    with c1:
        fig = base_map((ss.talhoes["lat"].mean(), ss.talhoes["lon"].mean()), 12.6, satellite)
        add_talhao_layer(fig, vals, colors, fmt)
        st.plotly_chart(fig, width="stretch")
        st.caption("Polígonos aproximados (quadrados com a área informada) quando não há GeoJSON importado.")

    rows = []
    for _, r in ss.talhoes.iterrows():
        n = str(r["nome"])
        rows.append({"Talhão": n, "Área (ha)": r["area_ha"], "Textura": r["textura"], "MO (%)": r["mo_pct"],
                     "pH": r["ph"], "Água disponível (mm/m)": AWC_MM_PER_M[r["textura"]],
                     "NDVI recente": ndvi_last[n], "NDVI médio 12 m": ndvi_mean[n],
                     "Vulnerabilidade": f"{vul[n][1]} ({vul[n][0]}/6)"})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

    series = [v.assign(Talhão=k) for k, v in ndvi.items() if v is not None]
    if series:
        s = pd.concat(series)
        fig = px.line(s, x="data", y="ndvi", color="Talhão", markers=True, color_discrete_sequence=PALETTE,
                      title="NDVI MODIS por talhão (últimos ~12 meses)")
        fig.add_hrect(y0=0, y1=0.3, fillcolor="#C62828", opacity=0.07, line_width=0,
                      annotation_text="solo exposto / palhada", annotation_position="bottom left")
        fig.add_hrect(y0=0.6, y1=1, fillcolor="#2E7D32", opacity=0.07, line_width=0,
                      annotation_text="vegetação vigorosa", annotation_position="top left")
        fig.update_layout(height=360, yaxis=dict(range=[0, 1]), margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("NDVI baixo na entressafra indica solo descoberto — oportunidade para culturas de cobertura. "
                   "Pixels de 250 m podem misturar talhões vizinhos, estradas e matas.")

    with st.expander("Resolução espacial: talhão × célula NASA POWER"):
        b = nasa_power.power_cell_bounds(ss.lat, ss.lon)
        fig = base_map((ss.lat, ss.lon), 7.5, satellite)
        fig.add_trace(go.Scattermap(lat=[b[0], b[0], b[1], b[1], b[0]], lon=[b[2], b[3], b[3], b[2], b[2]],
                                    mode="lines", line=dict(color="#FFD600", width=3), name="Célula NASA POWER"))
        fig.add_trace(go.Scattermap(lat=ss.talhoes["lat"], lon=ss.talhoes["lon"], mode="markers",
                                    marker=dict(size=9, color="#E53935"), name="Talhões"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Os dados climáticos representam a média de uma área de dezenas de quilômetros.")


# ----------------------------------------------------------------------------- 4. planejamento

def weights_controls() -> dict[str, float]:
    modes = list(WEIGHT_PRESETS) + ["Personalizado"]
    ss.weight_mode = st.radio("Prioridade", modes, index=modes.index(ss.weight_mode), horizontal=True)
    if ss.weight_mode == "Personalizado":
        cols = st.columns(5)
        for col, (k, label) in zip(cols, CRITERIA.items()):
            ss.custom_weights[k] = col.slider(label, 0, 100, int(round(ss.custom_weights.get(k, 0.2) * 100)), 5) / 100
    w = current_weights()
    total = sum(w.values()) or 1
    st.caption("Pesos: " + " · ".join(f"{CRITERIA[k]} {100 * v / total:.0f}%" for k, v in w.items()))
    return w


def render_option(i: int, row: pd.Series, t: Talhao, ref: pd.Series | None) -> None:
    exp = recommender.explain(row, t, ref)
    title = f"#{i + 1} · {row['Rotação']} — pontuação {row['pontuacao']:.0f}/100"
    with st.expander(title, expanded=i == 0):
        st.plotly_chart(rotation_timeline(row["rot"], row["delays"]), width="stretch", key=f"tl_{i}_{row.name}")
        c1, c2, c3 = st.columns([1.1, 1.3, 1.3])
        with c1:
            st.plotly_chart(criteria_bar(row), width="stretch", key=f"cb_{i}_{row.name}")
            st.metric("💰 Margem média (2 anos)", brl(row["margem_media"]) + "/ha",
                      help="Média nos anos históricos; P10–P90 abaixo.")
            st.caption(esc(f"P10–P90: {brl(row['margem_p10'])} a {brl(row['margem_p90'])} · "
                           f"água: {row['demanda_mm']:.0f} mm (+{row['irrig_mm']:.0f} irrigação)"))
        with c2:
            st.markdown("**Por que esta rotação**")
            for p in exp["pros"] or ["—"]:
                st.markdown(esc(f"- {p}"))
        with c3:
            st.markdown("**Riscos e cuidados**")
            for c in exp["cons"] or ["Nenhum risco relevante identificado pelas regras do protótipo."]:
                st.markdown(esc(f"- {c}"))
            if exp["comparacao"]:
                st.markdown("**Comparada à rotação atual**")
                st.markdown(esc(" · ".join(exp["comparacao"])))


def page_planejamento() -> None:
    hero("4. Planejamento de rotação", "Centenas de rotações simuladas no clima histórico real; veja as melhores "
         "para as suas prioridades e **por que** foram indicadas.")
    t = select_talhao()
    if t is None:
        return
    st.caption(f"Solo {t.textura}, pH {t.ph}, MO {t.mo_pct}% · {'irrigado' if t.irrigado else 'sequeiro'} · "
               f"rotação atual: {recommender.rotation_short(t.rotacao_atual)}")
    with st.container(border=True):
        options = [k for k in CROPS if k != "pousio"]
        sel = st.multiselect("Culturas que você aceita cultivar", options, default=[k for k in ss.allowed if k in options],
                             format_func=lambda k: CROPS[k].nome)
        if not any(k in SUMMER for k in sel):
            st.warning("Selecione ao menos uma cultura de verão.")
            return
        ss.allowed = sel
        w = weights_controls()
        c1, c2 = st.columns(2)
        use_min = c1.checkbox("Exigir margem mínima")
        min_margin = c2.number_input("Margem média mínima em 2 anos (R$/ha)", value=3000, step=500, disabled=not use_min)

    res = evaluate_talhao(t)
    if res.empty:
        st.error("Nenhuma rotação viável com as culturas selecionadas.")
        return
    ranked = recommender.rank(res, w)
    ref_rows = ranked[ranked["rot"] == t.rotacao_atual]
    ref = ref_rows.iloc[0] if not ref_rows.empty else None
    pool = ranked[ranked["margem_media"] >= min_margin] if use_min else ranked
    top = recommender.top_unique(pool, 5)
    st.subheader("Alternativas recomendadas")
    st.caption(f"{len(res)} rotações viáveis avaliadas com {int(res['n_anos'].iloc[0])} anos históricos NASA POWER. "
               "Cada alternativa foi simulada em todos os anos da série (balanço hídrico diário FAO-56).")
    if top.empty:
        st.warning("Nenhuma rotação atende à margem mínima.")
    for i, row in top.iterrows():
        render_option(i, row, t, ref)
    if ref is not None:
        pos = int(ranked.index[ranked["rot"] == t.rotacao_atual][0]) + 1
        with st.container(border=True):
            st.markdown(f"**Sua rotação atual** — {ref['Rotação']}: pontuação {ref['pontuacao']:.0f}/100 "
                        f"(posição {pos} de {len(ranked)})")
            cols = st.columns(5)
            for col, (k, label) in zip(cols, CRITERIA.items()):
                col.metric(label, f"{ref[k]:.0f}")
    with st.expander("Tabela completa"):
        show = ranked[["Rotação", "pontuacao", *CRITERIA, "margem_media", "margem_p10", "demanda_mm", "irrig_mm",
                       "deficit_pct", "p_quebra", "cobertura_dias", "familias"]].round(1)
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button("Baixar CSV", show.to_csv(index=False).encode("utf-8"), f"rotacoes_{t.nome}.csv")
    with st.expander("Como a pontuação é calculada"):
        st.markdown("""
- **Água**: déficit hídrico das culturas comerciais (ETa/ETc no balanço diário) e consumo total de água (ETc + irrigação).
- **Solo**: dias de cobertura (cultura + palhada), palhada, diversidade de famílias botânicas, leguminosas,
  raízes profundas e regras de sucessão. Coberturas com pouca água têm a contribuição reduzida.
- **Fitossanidade**: regras verificáveis de sucessão (mesma espécie/família, hospedeiros comuns de mofo-branco,
  nematoide-de-cisto, giberela no trigo após milho, crotalária para nematoides).
- **Risco climático**: frequência de quebra (>30% de perda) nas culturas comerciais em todos os anos históricos,
  considerando déficit hídrico (Ky FAO-33), calor no florescimento e geada.
- **Economia**: margem média e margem no pior decil (P10), normalizadas entre as alternativas.

Pesos definem o cenário (A, B, C ou personalizado). Produtividades, preços e custos vêm da tela 1.
""")


# ----------------------------------------------------------------------------- 5. comparação

def strategy_set(t: Talhao, res: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for name, w in WEIGHT_PRESETS.items():
        out[name.split(" — ")[0] + " · " + name.split(" — ")[1]] = recommender.top_unique(recommender.rank(res, w), 1).iloc[0]
    cur = res[res["rot"] == t.rotacao_atual]
    if not cur.empty:
        out["Atual"] = cur.iloc[0]
    return out


def page_comparacao() -> None:
    hero("5. Comparação de cenários de prioridade", "Qual rotação vence quando a prioridade muda? Compare "
         "água, solo, risco e margem lado a lado.")
    t = select_talhao()
    if t is None:
        return
    res = evaluate_talhao(t)
    if res.empty:
        return
    strategies = strategy_set(t, res)
    st.write("Melhor rotação para cada prioridade (A: água, B: solo, C: equilíbrio) comparada à rotação atual.")

    cols = st.columns(len(strategies))
    for col, (name, row) in zip(cols, strategies.items()):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            st.write(row["Rotação"])
            st.metric("💰 Margem média 2 anos", brl(row["margem_media"]))
            st.caption(f"Água {row['demanda_mm'] + row['irrig_mm']:.0f} mm · cobertura {row['cobertura_dias']:.0f} d · "
                       f"quebra {row['p_quebra']:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        labels = list(CRITERIA.values())
        for i, (name, row) in enumerate(strategies.items()):
            vals = [row[k] for k in CRITERIA]
            fig.add_scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name=name,
                                 line=dict(color=PALETTE[i], width=3, dash="dot" if name == "Atual" else "solid"))
        fig.update_layout(title="Indicadores (0–100)", polar=dict(radialaxis=dict(range=[0, 100])), height=420,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with c2:
        box = pd.concat([row["samples"][["margem"]].assign(Estratégia=name) for name, row in strategies.items()])
        fig = px.box(box, x="Estratégia", y="margem", color="Estratégia", points="all",
                     color_discrete_sequence=PALETTE, title="Margem em 2 anos em cada ano histórico (R$/ha)")
        fig.update_layout(height=420, showlegend=False, yaxis_title="R$/ha")
        st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame([{
        "Estratégia": n, "Rotação": r["Rotação"], "Margem média (R$/ha)": r["margem_media"],
        "Margem P10 (R$/ha)": r["margem_p10"], "Consumo de água (mm)": r["demanda_mm"] + r["irrig_mm"],
        "Déficit comerciais (%)": r["deficit_pct"], "Prob. quebra (%)": r["p_quebra"],
        "Cobertura (dias/730)": r["cobertura_dias"], "Famílias": r["familias"],
    } for n, r in strategies.items()]).round(0)
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Robustez da recomendação aos pesos")
    st.caption("Sorteamos 500 combinações aleatórias de pesos e contamos quantas vezes cada rotação fica em 1º lugar. "
               "Rotações que vencem com frequência são escolhas robustas mesmo com prioridades incertas.")
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(len(CRITERIA)), 500)
    scores = res[list(CRITERIA)].to_numpy() @ weights.T
    winners = res["Rotação"].iloc[scores.argmax(axis=0)].value_counts().head(6)
    fig = px.bar(x=winners.values / 5, y=winners.index, orientation="h", labels={"x": "% das combinações", "y": ""},
                 color_discrete_sequence=[PALETTE[1]])
    fig.update_layout(height=280, yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- 6. simulador

SIM_DEFAULTS = {"sim_p": 0, "sim_t": 0.0, "sim_atraso": 0, "sim_veranico": 0, "sim_mes": 1, "sim_calor": 0.0,
                "sim_irrig": 400}
SIM_PRESETS = {
    "Seca moderada": {"sim_p": -20, "sim_t": 1.0},
    "Seca severa": {"sim_p": -40, "sim_t": 2.0, "sim_atraso": 20},
    "Veranico em janeiro": {"sim_veranico": 20, "sim_mes": 1},
    "Aquecimento +2 °C": {"sim_t": 2.0, "sim_calor": 1.5},
}


def set_sim(values: dict) -> None:
    for k, v in {**SIM_DEFAULTS, **values}.items():
        ss[k] = v


def page_simulador() -> None:
    hero("6. Simulador de cenários climáticos", "Aplique seca, calor, atraso das chuvas e veranicos sobre os "
         "anos históricos e veja quais rotações resistem melhor.")
    st.info("**Cenário hipotético (análise “e se”)**: as alterações abaixo são aplicadas sobre os anos históricos "
            "da NASA POWER para testar a sensibilidade das rotações. **Não é previsão do tempo** (dias a semanas) "
            "**nem projeção climática** (décadas, modelos CMIP6/NEX-GDDP).")
    for k, v in SIM_DEFAULTS.items():
        ss.setdefault(k, v)
    t = select_talhao()
    if t is None:
        return
    cols = st.columns(len(SIM_PRESETS) + 1)
    for col, (name, vals) in zip(cols, SIM_PRESETS.items()):
        col.button(name, on_click=set_sim, args=(vals,), width="stretch")
    cols[-1].button("Restaurar", on_click=set_sim, args=({},), width="stretch")

    with st.container(border=True):
        a, b, c = st.columns(3)
        a.slider("Variação da chuva (%)", -60, 30, step=5, key="sim_p")
        a.slider("Aumento de temperatura (°C)", 0.0, 4.0, step=0.5, key="sim_t")
        b.slider("Atraso no início das chuvas (dias)", 0, 45, step=5, key="sim_atraso")
        b.slider("Veranico: dias sem chuva", 0, 40, step=5, key="sim_veranico")
        b.selectbox("Mês do veranico", list(range(1, 13)), format_func=lambda m: diagnostics.MONTHS[m - 1], key="sim_mes")
        c.slider("Extremos: +°C nos 10% dias mais quentes", 0.0, 4.0, step=0.5, key="sim_calor")
        c.slider("Água disponível para irrigação (mm/safra)", 0, 800, step=50, key="sim_irrig",
                 disabled=not t.irrigado, help="Aplica-se apenas a talhões irrigados (limite de outorga/reservatório).")
    sc = Scenario(ss.sim_p, ss.sim_t, ss.sim_atraso, ss.sim_veranico, ss.sim_mes, ss.sim_calor, float(ss.sim_irrig))
    base_sc = Scenario(limite_irrigacao_mm=float(ss.sim_irrig))

    res = evaluate_talhao(t, base_sc)
    if res.empty:
        return
    strategies = strategy_set(t, res)
    pool = recommender.top_unique(recommender.rank(res, current_weights()), 12)
    options = {r["Rotação"]: r["rot"] for _, r in pool.iterrows()}
    options.update({r["Rotação"]: r["rot"] for r in strategies.values()})
    default = list(dict.fromkeys(r["Rotação"] for r in strategies.values()))
    chosen = st.multiselect("Rotações comparadas", list(options), default=default)
    if not chosen:
        return
    rots = tuple(options[c] for c in chosen)
    base = evaluate_talhao(t, base_sc, rots).set_index("Rotação").loc[chosen]
    scen = evaluate_talhao(t, sc, rots).set_index("Rotação").loc[chosen]

    st.markdown(f"**Cenário:** {sc.describe()}")
    rows = []
    for name in chosen:
        b_, s_ = base.loc[name], scen.loc[name]
        rows.append({"Rotação": name, "Base": b_["margem_media"], "Cenário": s_["margem_media"],
                     "Base P10": b_["margem_p10"], "Cenário P10": s_["margem_p10"],
                     "Base P90": b_["margem_p90"], "Cenário P90": s_["margem_p90"],
                     "Quebra base": b_["p_quebra"], "Quebra cenário": s_["p_quebra"],
                     "Déficit base": b_["deficit_pct"], "Déficit cenário": s_["deficit_pct"],
                     "Anos com quebra": f"{int(s_['anos_quebra'])}/{int(s_['n_anos'])}"})
    comp = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for label, col, color in [("Clima histórico", "Base", "#90A4AE"), ("Cenário", "Cenário", "#E8710A")]:
            fig.add_bar(name=label, x=comp["Rotação"], y=comp[col], marker_color=color,
                        error_y=dict(type="data", symmetric=False, array=comp[f"{col} P90"] - comp[col],
                                     arrayminus=comp[col] - comp[f"{col} P10"]))
        fig.update_layout(barmode="group", title="Margem em 2 anos (média e faixa P10–P90, R$/ha)", height=420,
                          legend=dict(orientation="h", y=-0.35), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(name="Clima histórico", x=comp["Rotação"], y=comp["Quebra base"], marker_color="#90A4AE")
        fig.add_bar(name="Cenário", x=comp["Rotação"], y=comp["Quebra cenário"], marker_color="#C62828")
        fig.update_layout(barmode="group", title="Probabilidade de quebra (>30%) por cultura comercial (%)",
                          height=420, legend=dict(orientation="h", y=-0.35), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown("**Interpretação**")
        if sc.is_baseline():
            st.write("Sem alterações aplicadas: resultados iguais ao clima histórico. Ajuste os controles acima.")
        for _, r in comp.iterrows():
            delta = r["Cenário"] - r["Base"]
            rel = 100 * delta / abs(r["Base"]) if r["Base"] else 0
            st.markdown(esc(f"- **{r['Rotação']}**: margem média {brl(r['Cenário'])}/ha "
                            f"({'+' if delta >= 0 else '−'}{abs(rel):.0f}% vs. histórico); em {r['Anos com quebra']} anos "
                            f"simulados houve ao menos uma quebra de safra; pior decil {brl(r['Cenário P10'])}."))
        best = comp.sort_values("Cenário P10", ascending=False).iloc[0]
        st.success(f"Mais resiliente neste cenário (maior margem no pior decil): **{best['Rotação']}**.")

    detail = []
    for name in chosen:
        s_b, s_s = base.loc[name, "samples"], scen.loc[name, "samples"]
        rot = options[name]
        for i, key in enumerate(rot):
            if CROPS[key].comercial:
                detail.append({"Rotação": name, "Época": SLOT_LABELS[i], "Cultura": CROPS[key].nome,
                               "Água atendida — histórico": s_b[f"sat_{i}"].mean(),
                               "Água atendida — cenário": s_s[f"sat_{i}"].mean()})
    if detail:
        d = pd.DataFrame(detail).drop(columns="Época").groupby(["Rotação", "Cultura"], sort=False).mean().reset_index()
        fig = px.bar(d.melt(id_vars=["Rotação", "Cultura"], var_name="Clima", value_name="ETa/ETc"),
                     x="Cultura", y="ETa/ETc", color="Clima", barmode="group", facet_col="Rotação", facet_col_wrap=2,
                     facet_row_spacing=0.18,
                     color_discrete_sequence=["#90A4AE", "#4C8BF5"], title="Fração da demanda hídrica atendida")
        fig.update_layout(height=320 * ((len(chosen) + 1) // 2))
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Mapa de risco da propriedade neste cenário")
    st.caption("Probabilidade de quebra das culturas comerciais da rotação atual de cada talhão.")
    risk = {}
    for _, r in ss.talhoes.iterrows():
        tt = talhao_obj(r)
        e = evaluate_talhao(tt, sc, (tt.rotacao_atual,))
        e = e[e["rot"] == tt.rotacao_atual]
        risk[tt.nome] = float(e["p_quebra"].iloc[0]) if not e.empty else np.nan
    fig = base_map((ss.talhoes["lat"].mean(), ss.talhoes["lon"].mean()), 12.6, True)
    add_talhao_layer(fig, risk, color_scale(risk, "RdYlGn", reverse=True, vmin=0, vmax=60),
                     lambda v: "—" if v is None or np.isnan(v) else f"{v:.0f}% de quebra")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- 7. relatório

def build_report() -> tuple[str, pd.DataFrame]:
    df, meta = climate()
    lines = [f"# Relatório RotaClima — {farm_name()}", "",
             f"Gerado em {dt.date.today():%d/%m/%Y}. Localização: lat {ss.lat:.4f}, lon {ss.lon:.4f}.", "",
             "## Dados utilizados", "",
             f"- NASA POWER (diário, comunidade AG): {meta['inicio']:%d/%m/%Y} a {meta['fim']:%d/%m/%Y} "
             f"({meta['origem']}). Parâmetros: {', '.join(nasa_power.PARAMETERS)}.",
             "- NDVI MODIS MOD13Q1 (ORNL DAAC), quando disponível em cache.",
             ("- **EXEMPLO**: talhões, solo, rotação atual e valores econômicos fictícios." if ss.exemplo else
              "- Talhões, solo, rotação atual, preços e custos informados pelo produtor."),
             f"- Calibração da chuva: {rain_source() or 'não aplicada (chuva da NASA POWER sem ajuste)'}.",
             (f"- Produtividade colhida informada: {len(ss.yields)} registros." if ss.yields is not None else
              "- Produtividade colhida: não informada."), "",
             "## Diagnóstico climático", ""]
    lines += [f"- {x}" for x in diagnostics.summary_text(climate_adjusted(), ss.lat)]
    table = []
    for _, r in ss.talhoes.iterrows():
        t = talhao_obj(r)
        res = evaluate_talhao(t)
        if res.empty:
            continue
        strat = strategy_set(t, res)
        ref = strat.get("Atual")
        lines += ["", f"## {t.nome}", "",
                  f"{t.area_ha:.0f} ha · solo {t.textura} · pH {t.ph} · MO {t.mo_pct}% · "
                  f"{'irrigado' if t.irrigado else 'sequeiro'}", "",
                  "| Estratégia | Rotação (2 anos) | Margem média (R$/ha) | Água (mm) | Cobertura (dias) | Prob. quebra |",
                  "|---|---|---:|---:|---:|---:|"]
        for name, row in strat.items():
            lines.append(f"| {name} | {row['Rotação']} | {row['margem_media']:,.0f} | "
                         f"{row['demanda_mm'] + row['irrig_mm']:.0f} | {row['cobertura_dias']:.0f} | {row['p_quebra']:.0f}% |")
            table.append({"Talhão": t.nome, "Estratégia": name, "Rotação": row["Rotação"],
                          **{CRITERIA[k]: round(row[k], 1) for k in CRITERIA},
                          "Margem média": round(row["margem_media"]), "Margem P10": round(row["margem_p10"]),
                          "Água (mm)": round(row["demanda_mm"] + row["irrig_mm"]),
                          "Cobertura (dias)": round(row["cobertura_dias"]), "Prob. quebra (%)": round(row["p_quebra"], 1)})
        best = strat[list(strat)[2]]
        exp = recommender.explain(best, t, ref)
        lines += ["", f"**Justificativa (estratégia C — {best['Rotação']}):**", ""]
        lines += [f"- Vantagem: {p}" for p in exp["pros"]] + [f"- Atenção: {c}" for c in exp["cons"]]
        if exp["comparacao"]:
            lines += ["", "Comparada à rotação atual: " + "; ".join(exp["comparacao"]) + "."]
    lines += ["", "## Limitações", "",
              "- Clima de reanálise com resolução de dezenas de km; todos os talhões compartilham o mesmo clima.",
              "- Balanço hídrico simplificado (coeficiente único FAO-56, sem escoamento superficial e sem ascensão capilar).",
              "- Perdas por calor e geada usam fatores heurísticos não calibrados localmente.",
              "- Preços e custos não alterados na tela 1 são valores de referência, não da fazenda.",
              "- Regras fitossanitárias simplificadas; não substituem diagnóstico de pragas, doenças e nematoides.", "",
              "## Dados necessários para validação em campo", "",
              "- Análise de solo completa (química, granulometria) e, se possível, análise nematológica.",
              "- Histórico de produtividade por talhão e registros de pluviômetro da propriedade.",
              "- Custos reais de produção e preços regionais.",
              "- Consulta ao ZARC (MAPA) para janelas de semeadura e cultivares indicadas.",
              "- Acompanhamento por engenheiro agrônomo responsável.", "", f"_{DISCLAIMER}_"]
    return "\n".join(lines), pd.DataFrame(table)


def page_relatorio() -> None:
    hero("7. Relatório",
         "Resumo das estratégias por talhão, indicadores utilizados e dados necessários para validação em campo.")
    if climate()[0] is None:
        return
    report, table = build_report()
    c1, c2 = st.columns(2)
    c1.download_button("Baixar relatório (Markdown)", report.encode("utf-8"), "relatorio_rotaclima.md", width="stretch")
    c2.download_button("Baixar indicadores (CSV)", table.to_csv(index=False).encode("utf-8"),
                       "indicadores_rotaclima.csv", width="stretch")
    with st.container(border=True):
        st.markdown(report)


def page_sem_fazenda() -> None:
    hero(ss.page, "Esta tela usa os dados da sua fazenda, que ainda não foram informados.")
    st.info("Para continuar, informe na tela **1. Propriedade** a localização da fazenda e pelo menos um talhão "
            "(ou abra o arquivo da fazenda salvo em uma visita anterior).")
    st.button("Ir para 1. Propriedade", type="primary", on_click=lambda: ss.update(page=PAGES[0]))


if ss.page != PAGES[0] and not farm_ready():
    page_sem_fazenda()
    st.stop()

{
    PAGES[0]: page_propriedade, PAGES[1]: page_diagnostico, PAGES[2]: page_mapa, PAGES[3]: page_planejamento,
    PAGES[4]: page_comparacao, PAGES[5]: page_simulador, PAGES[6]: page_relatorio,
}[ss.page]()
