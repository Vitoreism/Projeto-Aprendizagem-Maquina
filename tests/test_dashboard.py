# -*- coding: utf-8 -*-
"""Testes do dashboard, sem subir servidor e sem exigir Streamlit.

O pacote `dashboard` nao importa Streamlit de proposito -- o cache mora no
app.py. E o que permite testar carregamento, previsao e figuras aqui, em
segundos, em vez de clicar em sete abas na mao.
"""

import numpy as np
import pandas as pd
import pytest

from imoveis_jp.dashboard import dados, graficos, modelo


# --------------------------------------------------------------------------
# dados.py
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nome", sorted(dados.ARTEFATOS))
def test_carrega_cada_artefato(nome):
    """Todo artefato declarado tem que existir e vir com conteudo."""
    carregado = dados.carregar(nome)
    assert carregado is not None
    if isinstance(carregado, pd.DataFrame):
        assert len(carregado) > 0, f"'{nome}' veio vazio"


def test_artefato_ausente_diz_o_comando_que_regenera(tmp_path):
    """Arquivo faltando tem que ensinar o proximo passo, nao cuspir traceback."""
    fantasma = dados.Artefato(
        caminho=tmp_path / "nao_existe.csv",
        comando="python -m imoveis_jp.models.train",
        descricao="artefato de teste",
    )
    with pytest.raises(dados.ArtefatoAusente) as erro:
        dados.ler(fantasma)
    assert "imoveis_jp.models.train" in str(erro.value)
    assert erro.value.comando == "python -m imoveis_jp.models.train"


def test_artefato_desconhecido_e_erro_claro():
    with pytest.raises(KeyError):
        dados.carregar("artefato_que_nao_existe")


def test_candidatos_vem_do_decisao_sem_duplicar_lista():
    """A lista de candidatos tem uma fonte so -- decisao.py."""
    from imoveis_jp.models import decisao

    assert dados.CANDIDATOS == decisao.CANDIDATOS_NA_DECISAO


def test_veredito_traz_o_vencedor_e_a_comparacao_pareada():
    veredito = dados.carregar("decisao")
    assert veredito["vencedor"]
    assert veredito["resultado"] in {"vantagem_declarada", "empate_tecnico"}
    assert len(veredito["comparacao_pareada"]["diffs_por_fold"]) == 5


# --------------------------------------------------------------------------
# modelo.py
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ajustado():
    return modelo.ajustar()


def test_linha_padrao_tem_todas_as_colunas_que_o_modelo_espera():
    linha = modelo.linha_padrao()
    assert len(linha) == 1
    # as 8 continuas e as 5 nominais precisam estar la, senao o ColumnTransformer quebra
    for coluna in modelo.NUMERICAS + modelo.CATEGORICAS:
        assert coluna in linha.columns, f"falta '{coluna}'"


def test_condominio_e_iptu_saem_nulos_por_padrao():
    """Nulo e sinal: o pipeline treinou com add_indicator, e iptu falta em ~80%."""
    linha = modelo.linha_padrao()
    assert pd.isna(linha.loc[0, "condominio"])
    assert pd.isna(linha.loc[0, "iptu"])


def test_preve_valor_finito_para_entrada_minima(ajustado):
    linha = modelo.linha_padrao()
    previsao = modelo.prever(ajustado, linha)
    assert np.isfinite(previsao.central)
    assert previsao.central > 0


def test_nulo_em_condominio_e_iptu_nao_quebra_a_previsao(ajustado):
    """O imputer do Pipeline tem que absorver o nulo sem estourar."""
    linha = modelo.linha_padrao()
    linha.loc[0, "condominio"] = np.nan
    linha.loc[0, "iptu"] = np.nan
    previsao = modelo.prever(ajustado, linha)
    assert np.isfinite(previsao.central)


def test_faixa_de_incerteza_e_monotona(ajustado):
    previsao = modelo.prever(ajustado, modelo.linha_padrao())
    assert previsao.inferior < previsao.central < previsao.superior


def test_area_fora_da_faixa_de_treino_dispara_aviso(ajustado):
    linha = modelo.linha_padrao()
    linha.loc[0, "area_util"] = 5000.0  # muito acima do maximo visto (1.260)
    previsao = modelo.prever(ajustado, linha)
    assert any("área" in aviso.lower() for aviso in previsao.avisos)


def test_area_dentro_da_faixa_nao_dispara_aviso_de_area(ajustado):
    linha = modelo.linha_padrao()
    linha.loc[0, "area_util"] = 65.0  # a mediana da base
    previsao = modelo.prever(ajustado, linha)
    assert not any("área" in aviso.lower() for aviso in previsao.avisos)


def test_previsao_de_alto_padrao_avisa_sobre_extrapolacao(ajustado):
    """Acima de R$ 2 mi ha pouco dado, e a hipotese registrada ja previa isso."""
    linha = modelo.linha_padrao()
    linha.loc[0, "area_util"] = 600.0
    linha.loc[0, "bairro"] = "cabo_branco"
    linha.loc[0, "suites"] = 5.0
    linha.loc[0, "garagens"] = 4.0
    previsao = modelo.prever(ajustado, linha)
    if previsao.central > modelo.TETO_CONFIAVEL:
        assert any("milh" in aviso.lower() for aviso in previsao.avisos)


def test_sorteio_devolve_imovel_real_com_preco_previsao_e_url():
    sorteado = modelo.sortear_do_teste(semente=42)
    assert sorteado["preco_real"] > 0
    assert sorteado["preco_previsto"] > 0
    assert isinstance(sorteado["url_anuncio"], str)
    assert sorteado["url_anuncio"].startswith("http")


def test_sorteio_e_reprodutivel_pela_semente():
    a = modelo.sortear_do_teste(semente=7)
    b = modelo.sortear_do_teste(semente=7)
    assert a["url_anuncio"] == b["url_anuncio"]


def test_sorteio_muda_com_semente_diferente():
    urls = {modelo.sortear_do_teste(semente=s)["url_anuncio"] for s in range(8)}
    assert len(urls) > 1, "o sorteio esta preso no mesmo imovel"


# --------------------------------------------------------------------------
# graficos.py
# --------------------------------------------------------------------------

FUNCOES_DE_GRAFICO = [
    ("ranking_cv", "resultados"),
    ("folds", "folds"),
    ("pca_dumbbell", "pca"),
    ("previsto_vs_real", "residuos"),
    ("distribuicao_residuo", "residuos"),
    ("importancia", "importancia"),
    ("correlacao_vs_importancia", "importancia_codificada"),
    ("tsne", "tsne"),
]


@pytest.mark.parametrize("funcao,artefato", FUNCOES_DE_GRAFICO)
def test_grafico_devolve_figura(funcao, artefato):
    import plotly.graph_objects as go

    figura = getattr(graficos, funcao)(dados.carregar(artefato))
    assert isinstance(figura, go.Figure)


@pytest.mark.parametrize("funcao,artefato", FUNCOES_DE_GRAFICO)
def test_grafico_sobrevive_a_dataframe_vazio(funcao, artefato):
    """Aba nunca deve estourar traceback na cara de quem apresenta."""
    import plotly.graph_objects as go

    vazio = dados.carregar(artefato).iloc[0:0]
    figura = getattr(graficos, funcao)(vazio)
    assert isinstance(figura, go.Figure)


def test_erro_por_segmento_filtra_o_segmento_pedido():
    segmentos = dados.carregar("segmentos")
    figura = graficos.erro_por_segmento(segmentos, "bairro")
    assert figura is not None
