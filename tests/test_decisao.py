# -*- coding: utf-8 -*-
"""Testes do criterio de decisao da Etapa 5 (issue #25).

O ponto e provar as duas condicoes do criterio isoladas uma da outra: sinal
consistente sem margem suficiente ainda e empate, e margem suficiente com um
sinal quebrado tambem e empate. So as duas juntas declaram vantagem.
"""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from imoveis_jp.models import decisao
from imoveis_jp.models.candidatos.base import Candidato


def _candidato(nome, grade=None):
    return Candidato(
        nome=nome,
        dono="teste",
        regressor=LinearRegression(),
        hipotese="hipotese sintetica so para o teste do criterio de decisao passar",
        grade=grade or {},
    )


def _folds(valores_por_modelo):
    """valores_por_modelo: {'a': [f0..f4], 'b': [f0..f4]}"""
    linhas = [
        {"modelo": nome, "fold": i, "mae_log": v}
        for nome, valores in valores_por_modelo.items()
        for i, v in enumerate(valores)
    ]
    return pd.DataFrame(linhas)


def test_vantagem_declarada_quando_todas_as_folds_concordam_e_margem_e_grande():
    fold_scores = _folds({
        "a": [0.20, 0.21, 0.19, 0.20, 0.205],
        "b": [0.22, 0.23, 0.21, 0.22, 0.225],
    })
    inscritos = {"a": _candidato("a"), "b": _candidato("b")}

    veredito = decisao.decidir(fold_scores, inscritos)

    assert veredito["resultado"] == "vantagem_declarada"
    assert veredito["vencedor"] == "a"
    assert veredito["comparacao_pareada"]["diferenca_media"] >= decisao.LIMIAR_VANTAGEM


def test_empate_tecnico_quando_uma_fold_discorda():
    fold_scores = _folds({
        "a": [0.20, 0.21, 0.19, 0.20, 0.230],  # ultima fold pior que 'b'
        "b": [0.22, 0.23, 0.21, 0.22, 0.225],
    })
    inscritos = {"a": _candidato("a"), "b": _candidato("b")}

    veredito = decisao.decidir(fold_scores, inscritos)

    assert veredito["resultado"] == "empate_tecnico"
    assert veredito["comparacao_pareada"]["todas_as_folds_a_favor"] is False


def test_empate_tecnico_quando_margem_fica_abaixo_do_limiar():
    fold_scores = _folds({
        "a": [0.2000, 0.2100, 0.1900, 0.2000, 0.2050],
        "b": [0.2020, 0.2120, 0.1920, 0.2020, 0.2070],  # diferenca de 0.002 < 0.005
    })
    inscritos = {"a": _candidato("a"), "b": _candidato("b")}

    veredito = decisao.decidir(fold_scores, inscritos)

    assert veredito["resultado"] == "empate_tecnico"
    assert veredito["comparacao_pareada"]["todas_as_folds_a_favor"] is True
    assert veredito["comparacao_pareada"]["diferenca_media"] < decisao.LIMIAR_VANTAGEM


def test_desempate_usa_explicabilidade_custo_e_numero_de_hiperparametros():
    empatados = ["mlp", "ols"]
    inscritos = {
        "mlp": _candidato("mlp", grade={"regressor__alpha": [1]}),
        "ols": _candidato("ols", grade={}),
    }
    ordem = decisao.desempate(empatados, inscritos)
    # ols e mais explicavel e mais barato que mlp nas tabelas hardcoded do modulo
    assert ordem[0] == "ols"


def test_recusa_candidato_fora_do_registro():
    fold_scores = _folds({"fantasma": [0.2] * 5})
    with pytest.raises(KeyError):
        decisao.decidir(fold_scores, inscritos={})
