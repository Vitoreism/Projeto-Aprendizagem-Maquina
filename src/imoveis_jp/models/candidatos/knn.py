# -*- coding: utf-8 -*-
"""KNN -- regressao por vizinhos mais proximos (issue #22)."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from imoveis_jp.models.candidatos.base import Candidato

#: n_neighbors cobre o eixo principal do modelo. Com ~12k amostras de treino e
#: one-hot expandido, k muito pequeno memoriza ruido local; k grande aproxima
#: a media global e perde bairro/amenities. weights=distance e o outro eixo
#: util -- vizinhos mais proximos pesam mais sem mudar a geometria do k.
GRADE = {
    "regressor__n_neighbors": [5, 11, 21, 41],
    "regressor__weights": ["uniform", "distance"],
}

CANDIDATO = Candidato(
    nome="knn",
    dono="dev KNN",
    # 11: meio-termo antes da busca; impar evita empates triviais em weights=uniform.
    regressor=KNeighborsRegressor(n_neighbors=11, weights="distance"),
    grade=GRADE,
    hipotese=(
        "Deve ganhar do OLS/Ridge se o preco for localmente suave no espaco de "
        "features (vizinhos com area, bairro e comodidades parecidos tem precos "
        "parecidos), capturando interacoes sem modela-las. Espera-se CV MAE(log) "
        "entre Ridge (~0,25) e o boosting (~0,20): melhor que linear, ainda "
        "atras do boosting porque a alta dimensionalidade do one-hot dilui a "
        "distancia euclidiana. Se ficar perto ou pior que Ridge, a maldição da "
        "dimensionalidade esta matando a nocao de vizinhanca."
    ),
    escalar_binarias=True,
    justificativa_escala=(
        "KNN usa distancia euclidiana: sem padronizar as binarias, as continuas "
        "(variancia ~1 apos o StandardScaler do bloco num) dominam a distancia "
        "sobre amenities 0/1 (variancia tipica <0,25). Escalar as binarias "
        "iguala as variancias para o vizinho ser escolhido por todas as "
        "dimensoes, nao so por area/condominio."
    ),
)
