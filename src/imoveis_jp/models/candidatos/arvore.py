# -*- coding: utf-8 -*-
"""Modelo candidato: Arvore de Decisao (DecisionTreeRegressor).

A Arvore de Decisao e o modelo mais explicavel do conjunto e a base do Gradient
Boosting. Com `max_depth=None`, a arvore memoriza o treino inteiro (~12.820 linhas
e 132 colunas pos-one-hot), produzindo erro proximo de zero no treino e MAE alto
na validacao. A busca em grade avalia poda via `max_depth`, `min_samples_leaf` e
`ccp_alpha` para equilibrar bias e variancia.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from imoveis_jp.models import dataset
from imoveis_jp.models.candidatos.base import Candidato

GRADE = {
    "regressor__max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, None],
    "regressor__min_samples_leaf": [1, 2, 5, 10, 20, 50],
    "regressor__ccp_alpha": [0.0, 0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.005],
}

CANDIDATO = Candidato(
    nome="arvore_decisao",
    dono="dev A",
    regressor=DecisionTreeRegressor(random_state=dataset.SEMENTE),
    grade=GRADE,
    hipotese=(
        "DecisionTreeRegressor e o modelo mais explicavel do conjunto e serve como "
        "ponte conceitual para o Gradient Boosting. Espera-se que com max_depth=None "
        "ocorra overfitting severo (erro de treino proximo a zero e erro de CV "
        "elevado). A poda via max_depth, min_samples_leaf e ccp_alpha reduzira "
        "a variancia e melhorara a generalizacao, mas o modelo solo devera ficar "
        "atras do Gradient Boosting devido a falta de ensembling."
    ),
    # Arvore de decisao faz cortes por limiar em features individuais (if feature > threshold),
    # sendo totalmente invariante a escala de atributos continuos ou binarios.
    escalar_binarias=False,
)
