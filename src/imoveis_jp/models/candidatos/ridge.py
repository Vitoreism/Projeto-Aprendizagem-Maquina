# -*- coding: utf-8 -*-
"""Ridge -- regressao linear regularizada."""

from __future__ import annotations

from sklearn.linear_model import Ridge

from imoveis_jp.models import dataset
from imoveis_jp.models.candidatos.base import Candidato

#: varredura logaritmica ampla: com as features padronizadas, o alpha util pode
#: estar a ordens de magnitude do default 1,0, e passos menores nao mudariam a
#: escolha.
#:
#: A busca ja rodou e devolveu exatamente o default (0,2673 em todos os cinco
#: valores, ate a quarta casa). Isso nao e desperdicio: e a evidencia de que o
#: gargalo do linear aqui NAO e regularizacao, e forma funcional. Um modelo que
#: nao melhora com nenhum alpha esta limitado por nao conseguir representar
#: interacao, e a grade e o que prova isso.
GRADE = {
    "regressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
}

CANDIDATO = Candidato(
    nome="ridge",
    dono="dev E (integrador)",
    regressor=Ridge(alpha=1.0, random_state=dataset.SEMENTE),
    grade=GRADE,
    hipotese=(
        "Perde do boosting por uma margem grande, e a margem e o resultado: ela "
        "mede quanta interacao existe no problema. Preco de imovel depende de "
        "area COMBINADA com bairro -- 100 m2 no Cabo Branco e 100 m2 no Gramame "
        "nao sao o mesmo produto -- e o modelo aditivo nao representa isso. "
        "Espera-se CV entre 0,25 e 0,27, contra ~0,20 do boosting."
    ),
    # binarias sem escala nao atrapalham o Ridge: a penalidade e sobre o
    # coeficiente, e o coeficiente de uma binaria em 0/1 ja esta na escala do
    # alvo. Escalar mudaria a forca relativa da regularizacao entre binarias e
    # continuas sem que haja motivo para preferir uma das duas.
    escalar_binarias=False,
)
