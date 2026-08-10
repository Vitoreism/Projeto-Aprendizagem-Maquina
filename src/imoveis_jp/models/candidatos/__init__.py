# -*- coding: utf-8 -*-
"""Registro por descoberta dos modelos candidatos.

Cada modelo mora num arquivo proprio deste pacote e exporta uma constante
`CANDIDATO`. Ninguem edita uma lista central -- e por isso que cinco pessoas
podem trabalhar na mesma semana sem colidir. O motivo nao e teorico: quando
duas pessoas mexeram em `build_features.py` na mesma janela, quatro dos cinco
conflitos foram em arquivo gerado.

Para inscrever um modelo::

    # candidatos/arvore.py
    from imoveis_jp.models.candidatos.base import Candidato

    CANDIDATO = Candidato(
        nome="arvore_decisao",
        dono="dev A",
        regressor=DecisionTreeRegressor(random_state=dataset.SEMENTE),
        hipotese="Perde do boosting, mas mostra a curva de overfitting.",
        grade={"regressor__max_depth": [4, 8, 16, None]},
    )

Nada alem disso. `descobrir()` acha o arquivo sozinho.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from imoveis_jp.models.candidatos.base import Candidato

__all__ = ["Candidato", "descobrir"]


def descobrir() -> Dict[str, Candidato]:
    """Varre o pacote e devolve os candidatos, indexados por nome.

    Modulos que comecam com '_' sao ignorados, o que permite rascunho e
    utilitario no pacote sem inscreve-los na comparacao.

    Um erro de importacao NAO e silenciado: um candidato quebrado que sumisse
    da comparacao sem avisar seria descoberto so no dia da consolidacao, que e
    exatamente o que esta infraestrutura existe para evitar.
    """
    achados: Dict[str, Candidato] = {}
    origem: Dict[str, str] = {}

    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "base":
            continue

        modulo = importlib.import_module(f"{__name__}.{info.name}")
        candidato = getattr(modulo, "CANDIDATO", None)
        if candidato is None:
            continue

        if not isinstance(candidato, Candidato):
            raise TypeError(
                f"'{info.name}.CANDIDATO' e {type(candidato).__name__}, "
                f"esperado Candidato."
            )
        if candidato.nome in achados:
            raise ValueError(
                f"Dois candidatos com o nome '{candidato.nome}': "
                f"'{origem[candidato.nome]}' e '{info.name}'. O nome vira chave "
                f"nos resultados, entao um sobrescreveria o outro em silencio."
            )

        achados[candidato.nome] = candidato
        origem[candidato.nome] = info.name

    return achados
