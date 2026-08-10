# -*- coding: utf-8 -*-
"""O contrato que todo modelo candidato precisa cumprir.

Um Candidato e a declaracao completa de um modelo na comparacao: o que ele e, o
que se espera dele e o que ele precisa do pre-processamento. Nada disso mora em
lista compartilhada -- cada dev cria um arquivo e exporta uma constante, e o
descobridor acha sozinho. Sem lista, sem conflito de merge.

Por que `hipotese` e obrigatoria e nao um comentario: dois dos achados mais
uteis do projeto foram previsoes ERRADAS -- prevemos que corrigir o vazamento
estrutural custaria acuracia e ela melhorou, e prevemos ganho de 0,005 a 0,015
com `venda_direta` e veio 0,0010. Nenhum dos dois apareceria se a hipotese
tivesse sido escrita depois do resultado. O dataclass torna a regra executavel:
sem hipotese, o teste falha e o candidato nao entra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class Candidato:
    """Um modelo inscrito na comparacao da Etapa 5.

    Atributos:
        nome: identificador unico. Vira a chave em `resultados_modelos.csv`.
        dono: quem responde por ele, para a revisao saber a quem perguntar.
        regressor: instancia ja configurada, SEM pre-processamento -- o
            `Pipeline` e montado por quem consome, para que todos os candidatos
            compartilhem exatamente o mesmo preparo (regra 1 do protocolo).
        grade: espaco de busca do `GridSearchCV`. Chaves prefixadas com
            'regressor__' porque o estimador vira o passo 'regressor' do
            Pipeline. Vazio significa "sem busca", nao "esqueci".
        hipotese: o que se espera e por que, escrito ANTES da primeira execucao.
        escalar_binarias: ver `justificativa_escala`.
        justificativa_escala: por que este modelo precisa (ou nao) da escala nas
            binarias. Obrigatoria quando `escalar_binarias=True` -- ligar a
            opcao sem dizer por que e a mesma coisa que nao decidir.
    """

    nome: str
    dono: str
    regressor: BaseEstimator
    hipotese: str
    grade: Dict[str, List[Any]] = field(default_factory=dict)
    escalar_binarias: bool = False
    justificativa_escala: str = ""

    def __post_init__(self) -> None:
        if not self.nome or not self.nome.strip():
            raise ValueError("Candidato sem nome.")
        if not self.hipotese or not self.hipotese.strip():
            raise ValueError(
                f"Candidato '{self.nome}' sem hipotese. Ela e escrita antes de "
                f"rodar, nao depois -- ver o docstring deste modulo."
            )
        if self.escalar_binarias and not self.justificativa_escala.strip():
            raise ValueError(
                f"Candidato '{self.nome}' liga escalar_binarias sem justificar. "
                f"A opcao muda a geometria do espaco de atributos; quem liga diz por que."
            )
        invalidas = [k for k in self.grade if not k.startswith("regressor__")]
        if invalidas:
            raise ValueError(
                f"Candidato '{self.nome}': chaves de grade sem o prefixo "
                f"'regressor__': {invalidas}. O estimador e o passo 'regressor' "
                f"do Pipeline, entao o GridSearchCV nao acharia esses parametros."
            )
