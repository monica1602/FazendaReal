# RotaClima Fazenda Real — seus dados + dados da NASA

Versão do RotaClima para **fazendas reais**: o agricultor informa os dados da própria propriedade e recebe
recomendações de rotação de culturas explicáveis, com clima da **NASA POWER** e vegetação do satélite **MODIS**.

Protótipo para o desafio **NASA Space Apps Challenge 2026 — "Mudança no Campo: Adaptando Fazendas com Dados da NASA"**.

> Protótipo de apoio à decisão. As recomendações **não substituem** a avaliação de um engenheiro agrônomo,
> a análise de solo e o Zoneamento Agrícola de Risco Climático (ZARC).

## O que o produtor informa (tela 1)

| Dado | Como |
|---|---|
| Localização | latitude/longitude ou clique no mapa de satélite |
| Talhões | tabela, desenho no mapa ou arquivo GeoJSON/KML (Google Earth) |
| Análise de solo | argila (define a textura), pH, matéria orgânica, irrigação |
| Rotação atual | 2 anos de verão e 2ª safra por talhão |
| Pluviômetro | CSV `data;chuva_mm` diário ou mensal; ajusta a chuva da NASA POWER pelos meses em comum (mín. 6) |
| Produtividade colhida | CSV `talhao;safra;cultura;produtividade_sc_ha` (ou `_t_ha`, `_kg_ha`); a média pode virar a produtividade de referência |
| Preços e custos | tabela editável (começa com valores de referência) |

Clima (NASA POWER, 2001 até hoje) e NDVI (MODIS MOD13Q1, 250 m) são buscados automaticamente para a localização.
As telas 2 a 7 (diagnóstico, mapa, rotação, cenários, simulador e relatório) usam esses dados.

## Onde ficam os dados

O site **não guarda** dados no servidor. Tudo fica no navegador enquanto a página está aberta. O botão
**💾 Salvar arquivo da fazenda** baixa um `.json` (formato `fazendareal/1`, ver `rotaclima/farmfile.py`), que pode ser
aberto na próxima visita. As coordenadas são enviadas à NASA POWER e ao ORNL DAAC apenas para buscar clima e NDVI.

Há exemplos (5 fazendas fictícias) para conhecer o site sem dados próprios; eles aparecem marcados como **exemplo**.

## Como executar

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q
```

### Link público (Streamlit Community Cloud, gratuito)

1. Acesse [share.streamlit.io](https://share.streamlit.io) e entre com a conta do GitHub.
2. **Create app** → repositório `monica1602/FazendaReal`, branch `main`, arquivo principal `app.py` → **Deploy**.

## Fontes e método

- NASA POWER (diário, comunidade AG): T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN, WS2M, GWETROOT.
- MODIS MOD13Q1 NDVI (ORNL DAAC MODIS Web Service).
- ET0 Penman-Monteith FAO-56; Kc FAO-56; Ky FAO-33; água disponível por textura inspirada na FAO-56.
- Regras de sucessão heurísticas, ainda não validadas por agrônomo.
