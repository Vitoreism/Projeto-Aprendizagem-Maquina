# -*- coding: utf-8 -*-
"""Candidato MLP -- Multi-Layer Perceptron (Rede Neural Feed-Forward).

Redes neurais feed-forward sao indicadas para regressao nao-linear e classificacao.
Em dados tabulares de ~13 mil linhas com relacoes esparsas e estruturadas, espera-se
que a MLP tenha desempenho intermediario: superando o modelo linear regularizado
(Ridge), porem ficando atras de modelos de boosting (HistGradientBoosting), que
constroem particoes ortogonais mais eficientes no espaco de atributos tabulares.
"""

from __future__ import annotations

from sklearn.neural_network import MLPRegressor

from imoveis_jp.models import dataset
from imoveis_jp.models.candidatos.base import Candidato

#: Espaco de busca de hiperparametros para o GridSearchCV.
#: Inclui ativacoes (relu, tanh, logistic), solvers (adam, sgd),
#: taxas de aprendizado iniciais (incluindo taxas elevadas para analisar estabilidade),
#: e diferentes arquiteturas de camadas ocultas.
GRADE = {
    "regressor__hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "regressor__activation": ["relu", "tanh", "logistic"],
    "regressor__solver": ["adam", "sgd"],
    "regressor__learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    "regressor__alpha": [0.0001, 0.001, 0.01],
}

CANDIDATO = Candidato(
    nome="mlp",
    dono="dev (feat/MLP)",
    regressor=MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=dataset.SEMENTE,
    ),
    grade=GRADE,
    hipotese=(
        "A MLP captura interacoes nao-lineares entre area e localizacao sem "
        "necessitar de termos cruzados manuais, devendo superar o Ridge (CV ~0,25). "
        "Contudo, em dados tabulares (~13k linhas), o otimizador por gradiente "
        "sofre com esparsidade e fronteiras de decisao continuas, devendo "
        "ficar atras do Gradient Boosting (CV ~0,20). Espera-se CV MAE(log) entre 0,21 e 0,23."
    ),
    # A otimizacao via gradiente exige todas as features na mesma escala.
    # Variaveis binarias em 0/1 possuem variancia p*(1-p) << 1, degradando a escala do gradiente.
    escalar_binarias=True,
    justificativa_escala=(
        "Redes Neurais (MLP) utilizam otimizadores baseados em gradiente descendente. "
        "Features binarias nao-escaladas (0 ou 1) possuem variancia menor do que as "
        "continuas padronizadas (variancia=1.0), fazendo com que as variaveis continuas "
        "dominem indevidamente a magnitude dos gradientes e desestabilizem a convergencia."
    ),
)
