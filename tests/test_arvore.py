# -*- coding: utf-8 -*-
"""Testes especificos do candidato Árvore de Decisão (Issue #21)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from imoveis_jp.models import candidatos, dataset, train
from imoveis_jp.models.candidatos.arvore import CANDIDATO, GRADE


def test_arvore_registrada_no_pacote():
    """Garante que o candidato 'arvore_decisao' e descoberto automaticamente."""
    inscritos = candidatos.descobrir()
    assert "arvore_decisao" in inscritos
    candidato = inscritos["arvore_decisao"]
    assert candidato.nome == "arvore_decisao"
    assert candidato.dono == "dev A"
    assert isinstance(candidato.regressor, DecisionTreeRegressor)


def test_arvore_hipotese_e_reprodutibilidade():
    """Valida semente random_state e tamanho da hipotese."""
    assert CANDIDATO.regressor.random_state == dataset.SEMENTE
    assert len(CANDIDATO.hipotese.strip()) >= 40


def test_arvore_grade_chaves_e_bordas():
    """Garante que a grade contem as chaves corretas e hiperparametros de poda."""
    assert "regressor__max_depth" in GRADE
    assert "regressor__min_samples_leaf" in GRADE
    assert "regressor__ccp_alpha" in GRADE
    for k in GRADE:
        assert k.startswith("regressor__")


def test_arvore_fit_predict_pipeline():
    """Testa treino e predicao da Arvore dentro do Pipeline com dados de treino."""
    rng = np.random.default_rng(dataset.SEMENTE)
    n = 50
    X = pd.DataFrame(
        {
            "area_util": rng.uniform(40, 200, n),
            "quartos": rng.integers(1, 5, n).astype(float),
            "com_piscina": rng.integers(0, 2, n),
            "venda_direta": rng.integers(0, 2, n),
            "bairro": rng.choice(["bessa", "manaira", "gramame"], n),
        }
    )
    y = pd.Series(np.log(X["area_util"] * 8000 + 50_000))

    num, bin_, cat = dataset.colunas_por_tipo(X)
    pipe = Pipeline(
        [
            ("preparo", train.montar_preprocessador(num, bin_, cat, CANDIDATO.escalar_binarias)),
            ("regressor", CANDIDATO.regressor),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == n
    assert np.isfinite(preds).all()
