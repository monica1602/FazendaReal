from rotaclima import theme
from tests.test_app import PAGES


def test_every_page_has_distinct_theme_and_illustration():
    assert list(theme.THEMES) == PAGES
    assert len({t.primaria for t in theme.THEMES.values()}) == len(PAGES)
    for page, t in theme.THEMES.items():
        assert (theme.ASSETS / t.ilustracao).exists()
        assert len(set(t.paleta)) >= 5
        assert t.primaria in theme.css(page)
        assert theme.hero_image(page).startswith('<img src="data:image/svg+xml;base64,')
