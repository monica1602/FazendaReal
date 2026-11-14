"""Identidade visual: uma paleta e uma ilustração por tela, ligadas ao tema de cada página."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"


@dataclass(frozen=True)
class PageTheme:
    icone: str
    primaria: str
    escura: str
    clara: str
    destaque: str
    paleta: tuple[str, ...]
    ilustracao: str
    resumo: str


THEMES: dict[str, PageTheme] = {
    "1. Propriedade": PageTheme(
        "🚜", "#8D6E63", "#4E342E", "#EFEBE9", "#7CB342",
        ("#6D4C41", "#7CB342", "#C0A062", "#33691E", "#A1887F", "#D4A373", "#558B2F", "#BCAAA4"),
        "1_propriedade.svg", "Terra e lavoura: localização, talhões, solo e premissas da fazenda."),
    "2. Diagnóstico ambiental": PageTheme(
        "🌦️", "#1E88E5", "#0D47A1", "#E3F2FD", "#FFB300",
        ("#1E88E5", "#FB8C00", "#43A047", "#E53935", "#00ACC1", "#8E24AA", "#FDD835", "#5C6BC0"),
        "2_diagnostico.svg", "Água e clima: chuva, evaporação, calor e veranicos medidos pela NASA POWER."),
    "3. Mapa dos talhões": PageTheme(
        "🛰️", "#00897B", "#004D40", "#E0F2F1", "#9CCC65",
        ("#00897B", "#7CB342", "#FDD835", "#5C6BC0", "#26C6DA", "#8D6E63", "#EC407A", "#9E9D24"),
        "3_mapa.svg", "Olhar do satélite: vigor da vegetação (NDVI MODIS) e condição do solo de cada talhão."),
    "4. Planejamento de rotação": PageTheme(
        "🌱", "#43A047", "#1B5E20", "#E8F5E9", "#F9A825",
        ("#2E7D32", "#F9A825", "#8D6E63", "#66BB6A", "#FBC02D", "#00897B", "#AED581", "#D84315"),
        "4_planejamento.svg", "Sequências de cultivo para dois anos, com motivos, riscos e comparação com o atual."),
    "5. Comparação de cenários": PageTheme(
        "⚖️", "#5E35B1", "#311B92", "#EDE7F6", "#26A69A",
        ("#1E88E5", "#8D6E63", "#F9A825", "#78909C", "#7E57C2", "#26A69A", "#EC407A", "#9CCC65"),
        "5_comparacao.svg", "A: conservar água · B: recuperar o solo · C: equilíbrio econômico e ambiental."),
    "6. Simulador climático": PageTheme(
        "🔥", "#F4511E", "#BF360C", "#FBE9E7", "#FFB300",
        ("#F4511E", "#FFB300", "#6D4C41", "#1E88E5", "#8E24AA", "#43A047", "#D81B60", "#90A4AE"),
        "6_simulador.svg", "E se chover menos ou fizer mais calor? Teste a resistência de cada rotação."),
    "7. Relatório": PageTheme(
        "📋", "#0B3D91", "#061F4A", "#E8EEF8", "#FC3D21",
        ("#0B3D91", "#FC3D21", "#43A047", "#FFB300", "#00ACC1", "#8E24AA", "#6D4C41", "#90A4AE"),
        "7_relatorio.svg", "Síntese para apresentar e levar a campo: estratégias, justificativas e limitações."),
}


@lru_cache(maxsize=None)
def svg_uri(name: str) -> str:
    data = (ASSETS / name).read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


def nav_label(page: str) -> str:
    return f"{THEMES[page].icone}  {page}"


def css(page: str) -> str:
    t = THEMES[page]
    nav = "\n".join(
        f'section[data-testid="stSidebar"] [role="radiogroup"] > div:nth-of-type({i}) > label '
        f"{{ border-left: 5px solid {th.primaria}; }}\n"
        f'section[data-testid="stSidebar"] [role="radiogroup"] > div:nth-of-type({i}) > label:has(input:checked) '
        f"{{ background: {th.primaria}; }}"
        for i, th in enumerate(THEMES.values(), start=1)
    )
    return f"""
<style>
.stApp {{ background: linear-gradient(180deg, {t.clara} 0px, #FFFFFF 420px); }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {t.clara} 0%, #FFFFFF 70%);
  border-right: 4px solid {t.primaria}; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div > label {{ padding: 6px 10px; margin: 3px 0;
  border-radius: 8px; background: rgba(255,255,255,.75); transition: all .15s; width: 100%; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div > label:has(input:checked) p {{ color: #FFFFFF; font-weight: 700; }}
{nav}
.rc-brand {{ display: flex; align-items: center; gap: 10px; }}
.rc-brand b {{ font-size: 1.45rem; color: {t.escura}; }}
.st-key-hero {{ background: linear-gradient(120deg, {t.primaria} 0%, {t.escura} 100%); border-radius: 18px;
  padding: 1.1rem 1.6rem; box-shadow: 0 8px 22px rgba(0,0,0,.14); }}
.st-key-hero h1, .st-key-hero p, .st-key-hero span {{ color: #FFFFFF !important; }}
.st-key-hero h1 {{ padding-bottom: .2rem; }}
.st-key-hero .rc-chip {{ display: inline-block; background: {t.destaque}; color: #FFFFFF; border-radius: 999px;
  padding: 2px 12px; font-size: .8rem; font-weight: 700; letter-spacing: .03em; }}
.st-key-hero img {{ width: 100%; max-height: 170px; filter: drop-shadow(0 4px 8px rgba(0,0,0,.25)); }}
h2, h3 {{ color: {t.escura}; }}
[data-testid="stMainBlockContainer"] h3 {{ border-left: 6px solid {t.primaria}; padding-left: .6rem; }}
[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {t.clara}; border-left: 6px solid {t.primaria};
  border-radius: 12px; padding: .7rem .9rem; box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
[data-testid="stMetricValue"] {{ color: {t.escura}; font-size: 1.65rem; }}
[data-testid="stMetricLabel"] p {{ white-space: normal; font-weight: 600; color: {t.escura}; }}
.stButton button, .stDownloadButton button {{ border: 1.5px solid {t.primaria}; color: {t.escura}; border-radius: 10px; }}
.stButton button:hover, .stDownloadButton button:hover {{ background: {t.primaria}; color: #FFFFFF; border-color: {t.primaria}; }}
[data-testid="stExpander"] details {{ border-left: 5px solid {t.destaque}; border-radius: 10px; background: #FFFFFF; }}
[data-testid="stExpander"] summary:hover p {{ color: {t.primaria}; }}
span[data-baseweb="tag"] {{ background-color: {t.primaria} !important; }}
[data-testid="stSlider"] div[style*="translate(-50%"] {{ background-color: {t.primaria} !important; }}
[data-testid="stSliderThumbValue"] p {{ color: {t.escura} !important; font-weight: 700; }}
</style>
"""


def brand_html() -> str:
    return (f'<div class="rc-brand"><img src="{svg_uri("logo.svg")}" width="40"/>'
            f"<b>RotaClima</b><small style=\"display:block;font-size:.75rem\">Fazenda Real</small></div>")


def hero_image(page: str) -> str:
    t = THEMES[page]
    return f'<img src="{svg_uri(t.ilustracao)}" alt="Ilustração: {page}"/>'

