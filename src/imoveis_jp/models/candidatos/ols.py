# -*- coding: utf-8 -*-
"""OLS -- regressao linear ordinaria (sem regularizacao)."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from imoveis_jp.models.candidatos.base import Candidato

#: OLS nao tem hiperparametro util para buscar: `fit_intercept` e fixo em True
#: porque o Pipeline padroniza as continuas e o one-hot das nominais e completo
#: (menos a categoria de referencia implicita). Grade vazia = sem busca.
GRADE = {}

CANDIDATO = Candidato(
    nome="ols",
    dono="dev OLS",
    regressor=LinearRegression(),
    grade=GRADE,
    hipotese=(
        "Deve ficar muito perto do Ridge (CV MAE(log) ~0,25), porque a busca "
        "de alpha do Ridge nao melhorou o default -- sinal de que o gargalo "
        "do linear nao e regularizacao, e forma funcional. Espera-se perder "
        "do boosting pela mesma margem do Ridge: preco depende de interacoes "
        "(area x bairro) que o modelo aditivo nao representa. Se OLS for bem "
        "pior que Ridge, a colinearidade do one-hot esta inflando os coeficientes."
    ),
    escalar_binarias=False,
)
