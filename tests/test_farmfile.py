import numpy as np
import pandas as pd

from rotaclima import farmfile


def test_rain_csv_daily_and_calibration():
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    text = "data;chuva_mm\n" + "\n".join(f"{d:%d/%m/%Y};{'2,5' if d.day % 2 else '0'}" for d in days)
    rec = farmfile.read_rain_csv(text)
    assert len(rec) == len(days) and not rec["mensal"].any()
    power = pd.Series(1.0, index=days)
    cal = farmfile.rain_calibration(rec, power)
    assert cal["meses"] == 12
    assert np.isclose(cal["fator"], rec["chuva_mm"].sum() / len(days))


def test_rain_csv_monthly():
    rec = farmfile.read_rain_csv("mes,chuva_mm\n01/2024,250\n2024-02,180.5\n")
    assert rec["mensal"].all() and rec["chuva_mm"].tolist() == [250, 180.5]


def test_yield_csv_converts_sacks_and_recognizes_crops():
    y = farmfile.read_yield_csv("Talhão;Safra;Cultura;Produtividade (sc/ha)\nT1;2023/24;Soja;60\nT1;2024;Milho 2ª safra;100\n"
                                "T2;2024;Café;30\n")
    assert y["cultura"].tolist() == ["soja", "milho_2s"]
    assert np.allclose(y["produtividade_t_ha"], [3.6, 6.0])
    assert y.attrs["nao_reconhecidas"] == ["Café"]


def test_kml_polygon_area():
    kml = """<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Talhão A</name><Polygon>
    <outerBoundaryIs><LinearRing><coordinates>-51.0,-17.0,0 -50.99,-17.0,0 -50.99,-17.01,0 -51.0,-17.01,0 -51.0,-17.0,0
    </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>"""
    (name, poly), = farmfile.parse_kml(kml)
    lat, lon, area = farmfile.polygon_stats(poly)
    assert name == "Talhão A" and np.isclose(lat, -17.005) and 110 < area < 125


def test_textura_and_file_roundtrip():
    assert [farmfile.textura_from_argila(x) for x in (10, 20, 50)] == ["arenosa", "média", "argilosa"]
    talhao = {"nome": "T1", "lat": -17.0, "lon": -51.0, "area_ha": 10, "textura": "média", "ph": 5.5, "mo_pct": 2,
              "irrigado": False, "verao_1": "soja", "segunda_1": "milho_2s", "verao_2": "soja", "segunda_2": "trigo"}
    doc = farmfile.loads(farmfile.dumps({"propriedade": {"nome": "X", "lat": -17.0, "lon": -51.0}, "talhoes": [talhao]}))
    assert doc["talhoes"][0] == talhao
    for bad in ('{"a": 1}', farmfile.dumps({"talhoes": [{**talhao, "verao_1": "cafe"}]})):
        try:
            farmfile.loads(bad)
            raise AssertionError("deveria falhar")
        except ValueError:
            pass
