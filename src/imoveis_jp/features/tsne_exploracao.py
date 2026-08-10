# -*- coding: utf-8 -*-
"""t-SNE 2D colorido por faixa de preco -- EDA pura (issue #25).

Isto NAO entra em nenhum Pipeline de modelo: t-SNE nao tem `.transform` em dado
novo (cada chamada reprojeta o conjunto inteiro), entao nao serve como passo de
um `Pipeline` que precisa prever fora da amostra. O unico uso legitimo aqui e
visual -- ver se os imoveis se agrupam no espaco de atributos de um jeito que
acompanha o preco, antes/paralelamente a qualquer modelo.

Roda sobre a base inteira (nao so o treino): e leitura exploratoria, nao
avaliacao de modelo, entao a regra de nao tocar no teste nao se aplica aqui.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from imoveis_jp import config
from imoveis_jp.models import dataset, train

SAIDA_COORDS = config.PROCESSED / "tsne_coords.csv"
SAIDA_FIGURA = config.ROOT / "docs" / "figuras" / "tsne_precos.png"

ROTULOS_FAIXA = ["Q1 (mais barato)", "Q2", "Q3", "Q4", "Q5 (mais caro)"]


def faixas_de_preco(y_log: pd.Series) -> pd.Series:
    reais = np.exp(y_log.to_numpy())
    bordas = np.unique(np.quantile(reais, np.linspace(0, 1, len(ROTULOS_FAIXA) + 1)))
    rotulos = ROTULOS_FAIXA if len(bordas) == len(ROTULOS_FAIXA) + 1 else None
    return pd.cut(reais, bins=bordas, labels=rotulos, include_lowest=True)


def executar() -> pd.DataFrame:
    config.ensure_dirs()

    X, y, _ = dataset.carregar()
    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    # escalar_binarias=True: t-SNE e baseado em distancia euclidiana, mesmo
    # argumento do KNN/MLP em train.py -- sem isso as continuas dominam.
    preprocessador = train.montar_preprocessador(numericas, binarias, categoricas, escalar_binarias=True)
    Z = preprocessador.fit_transform(X)

    print(f"[tSNE] {Z.shape[0]} linhas x {Z.shape[1]} colunas -- pode demorar alguns minutos...", flush=True)
    projecao = TSNE(
        n_components=2,
        random_state=dataset.SEMENTE,
        init="pca",
        perplexity=30,
        n_jobs=-1,
    ).fit_transform(Z)

    tabela = pd.DataFrame(
        {
            "tsne_1": projecao[:, 0],
            "tsne_2": projecao[:, 1],
            "faixa_preco": faixas_de_preco(y).to_numpy(),
        }
    )
    tabela.to_csv(SAIDA_COORDS, index=False, encoding="utf-8")

    SAIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    cores = plt.cm.viridis(np.linspace(0, 1, len(ROTULOS_FAIXA)))
    fig, ax = plt.subplots(figsize=(9, 8))
    for cor, faixa in zip(cores, ROTULOS_FAIXA):
        pontos = tabela[tabela["faixa_preco"] == faixa]
        ax.scatter(pontos["tsne_1"], pontos["tsne_2"], s=6, alpha=0.5, color=cor, label=faixa)
    ax.set_title("t-SNE 2D dos imóveis, colorido por faixa de preço (EDA)")
    ax.set_xlabel("dimensão t-SNE 1")
    ax.set_ylabel("dimensão t-SNE 2")
    ax.legend(fontsize=8, markerscale=2, title="faixa de preço")
    fig.tight_layout()
    fig.savefig(SAIDA_FIGURA, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[tSNE] coordenadas: {SAIDA_COORDS}", flush=True)
    print(f"[tSNE] figura:      {SAIDA_FIGURA}", flush=True)
    return tabela


if __name__ == "__main__":
    executar()
