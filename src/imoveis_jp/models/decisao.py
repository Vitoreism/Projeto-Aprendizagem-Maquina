# -*- coding: utf-8 -*-
"""Criterio de decisao da Etapa 5 (issue #25) -- so le CV, nunca le teste.

Regras do criterio (declaradas ANTES de qualquer resultado, ver a issue #25):

1. Vence quem tiver o menor MAE medio na CV.
2. A vantagem sobre o segundo colocado so e DECLARADA se as duas condicoes
   valerem: a diferenca pareada favorece o mesmo modelo nas cinco folds, e a
   diferenca media e >= LIMIAR_VANTAGEM.
3. Se qualquer uma falhar: empate tecnico. Desempate nesta ordem --
   explicabilidade, custo de previsao, numero de hiperparametros.
4. O teste e avaliado depois, uma vez, e nao participa da decisao -- por isso
   este modulo nunca abre uma coluna '*_teste'.

Por que 0,005: o desvio entre folds das melhores configuracoes do projeto fica
entre 0,0032 e 0,0043 (ver docs/protocolo_comparacao.md). Diferenca menor que
um desvio nao se sustenta -- ja aconteceu de a busca de hiperparametros eleger
uma configuracao por 0,0005 quando o eixo inteiro nao tinha efeito.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from imoveis_jp import config
from imoveis_jp.models import candidatos
from imoveis_jp.models.candidatos.base import Candidato

SAIDA_CV_FOLDS = config.PROCESSED / "cv_mae_por_fold.csv"
SAIDA_DECISAO = config.PROCESSED / "decisao_criterio.json"

#: os seis candidatos que competem pela decisao -- baseline e gradient_boosting
#: padrao sao referencia, nao candidatos (protocolo_comparacao.md secao 3.1).
CANDIDATOS_NA_DECISAO = [
    "arvore_decisao",
    "knn",
    "mlp",
    "ols",
    "ridge",
    "gradient_boosting_ajustado",
]

LIMIAR_VANTAGEM = 0.005

#: 1 = mais explicavel. Julgamento registrado aqui, nao medido: arvore da a
#: regra em texto; OLS/Ridge dao um coeficiente por atributo; o boosting soma
#: centenas de arvores rasas (ainda auditavel via importancia); MLP e KNN nao
#: produzem nem regra nem coeficiente -- KNN "explica" so por analogia aos
#: vizinhos, e nem isso fica estavel em alta dimensao.
EXPLICABILIDADE = {
    "arvore_decisao": 1,
    "ols": 2,
    "ridge": 2,
    "gradient_boosting_ajustado": 3,
    "mlp": 4,
    "knn": 4,
}

#: 1 = mais barato por previsao nova. OLS/Ridge/Arvore respondem com uma
#: operacao fechada (produto interno ou uma descida na arvore); o boosting soma
#: a saida de cada arvore do ensemble; MLP e um produto de matrizes de tamanho
#: fixo mas maior; KNN precisa varrer (ou indexar) o treino inteiro a cada
#: previsao -- o unico cujo custo cresce com o tamanho da base.
CUSTO_PREVISAO = {
    "ols": 1,
    "ridge": 1,
    "arvore_decisao": 1,
    "gradient_boosting_ajustado": 2,
    "mlp": 2,
    "knn": 3,
}


def carregar_fold_scores(caminho=None) -> pd.DataFrame:
    caminho = caminho or SAIDA_CV_FOLDS
    if not caminho.exists():
        raise FileNotFoundError(
            f"'{caminho}' nao existe. Rode antes: python -m imoveis_jp.models.train"
        )
    return pd.read_csv(caminho)


def media_por_modelo(fold_scores: pd.DataFrame) -> pd.Series:
    return fold_scores.groupby("modelo")["mae_log"].mean().sort_values()


def comparar_pareado(fold_scores: pd.DataFrame, vencedor: str, desafiante: str) -> Dict:
    """diff = desafiante - vencedor, por fold. Positivo == vencedor melhor (MAE menor)."""
    v = fold_scores.loc[fold_scores["modelo"] == vencedor].sort_values("fold")["mae_log"].to_numpy()
    d = fold_scores.loc[fold_scores["modelo"] == desafiante].sort_values("fold")["mae_log"].to_numpy()
    if len(v) != len(d) or len(v) == 0:
        raise ValueError(
            f"'{vencedor}' tem {len(v)} folds e '{desafiante}' tem {len(d)} -- "
            f"nao da para parear."
        )
    diffs = d - v
    todas_a_favor = bool(np.all(diffs > 0))
    diferenca_media = float(diffs.mean())
    return {
        "vencedor": vencedor,
        "desafiante": desafiante,
        "diffs_por_fold": diffs.tolist(),
        "diferenca_media": diferenca_media,
        "todas_as_folds_a_favor": todas_a_favor,
        "vantagem_declarada": bool(todas_a_favor and diferenca_media >= LIMIAR_VANTAGEM),
    }


def desempate(nomes: List[str], inscritos: Dict[str, Candidato]) -> List[str]:
    """Ordena por explicabilidade, depois custo de previsao, depois numero de hiperparametros."""

    def chave(nome: str):
        return (
            EXPLICABILIDADE.get(nome, 99),
            CUSTO_PREVISAO.get(nome, 99),
            len(inscritos[nome].grade),
        )

    return sorted(nomes, key=chave)


def decidir(fold_scores: pd.DataFrame, inscritos: Dict[str, Candidato]) -> Dict:
    presentes = set(fold_scores["modelo"].unique())
    faltando = presentes - set(inscritos)
    if faltando:
        raise KeyError(f"modelo(s) em cv_mae_por_fold.csv sem Candidato registrado: {faltando}")

    medias = media_por_modelo(fold_scores)
    ranking = medias.index.tolist()
    vencedor_cv, segundo_cv = ranking[0], ranking[1]

    pareado = comparar_pareado(fold_scores, vencedor_cv, segundo_cv)

    if pareado["vantagem_declarada"]:
        resultado = "vantagem_declarada"
        vencedor_final = vencedor_cv
        ordem_desempate = None
    else:
        resultado = "empate_tecnico"
        ordem_desempate = desempate([vencedor_cv, segundo_cv], inscritos)
        vencedor_final = ordem_desempate[0]

    return {
        "resultado": resultado,
        "vencedor": vencedor_final,
        "ranking_cv": ranking,
        "medias_cv": medias.to_dict(),
        "comparacao_pareada": pareado,
        "ordem_desempate": ordem_desempate,
    }


def executar() -> Dict:
    config.ensure_dirs()
    fold_scores = carregar_fold_scores()
    fold_scores = fold_scores[fold_scores["modelo"].isin(CANDIDATOS_NA_DECISAO)]

    inscritos = candidatos.descobrir()
    veredito = decidir(fold_scores, inscritos)

    with open(SAIDA_DECISAO, "w", encoding="utf-8") as f:
        json.dump(veredito, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("CRITERIO DE DECISAO -- Etapa 5 (issue #25)")
    print("=" * 72)
    print("Ranking por CV MAE(log) medio:")
    for i, nome in enumerate(veredito["ranking_cv"], start=1):
        print(f"  {i}. {nome:28s} {veredito['medias_cv'][nome]:.4f}")
    p = veredito["comparacao_pareada"]
    print(
        f"\n{p['vencedor']} vs {p['desafiante']}: "
        f"diferenca media={p['diferenca_media']:.4f}, "
        f"todas as folds a favor={p['todas_as_folds_a_favor']}"
    )
    print(f"\nRESULTADO: {veredito['resultado'].upper()} -- vencedor: {veredito['vencedor']}")
    if veredito["ordem_desempate"]:
        print(f"Ordem de desempate: {veredito['ordem_desempate']}")
    print(f"\nSalvo em: {SAIDA_DECISAO}")

    return veredito


if __name__ == "__main__":
    executar()
