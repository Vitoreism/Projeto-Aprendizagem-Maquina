# -*- coding: utf-8 -*-
"""Figuras Plotly do dashboard. Funcoes puras: DataFrame entra, Figure sai.

DECISOES DE COR, e por que elas nao sao gosto pessoal:

- Modelo e categoria NOMINAL (nao ha ordem natural entre 'knn' e 'ridge'), entao
  colorir barra mais escura onde e maior seria gastar o canal de cor repetindo o
  que o comprimento da barra ja diz. Uma serie, uma cor.
- Onde a historia e "um venceu", a forma correta e ENFASE: o vencedor na cor
  cheia, o resto em cinza recessivo. Seis cores para contar um vencedor e o jeito
  mais comum de um grafico perder o proprio ponto.
- Faixa de preco (Q1..Q5) SIM tem ordem, entao usa rampa ordinal de um tom so,
  claro->escuro. Rampa multi-tom (arco-iris) em magnitude e erro.

As paletas foram validadas pelo script do metodo (nao no olho): o par
azul/laranja passa os cinco checks categoricos, e a rampa de cinco passos passa
os quatro checks ordinais, ambos na superficie clara.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# --- tokens da superficie clara (o app fixa o tema claro) --------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_MUDA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

#: slot 1 e 2 do tema categorico.
AZUL = "#2a78d6"
LARANJA = "#eb6834"

#: rampa ordinal de um tom so, claro->escuro. O passo mais claro (250) e o mais
#: claro que ainda separa da superficie (2,06:1).
RAMPA_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

FONTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: o vencedor do criterio -- recebe a cor cheia; o resto fica recessivo.
VENCEDOR = "gradient_boosting_ajustado"

ROTULOS = {
    "gradient_boosting_ajustado": "Gradient Boosting (ajustado)",
    "gradient_boosting": "Gradient Boosting (padrão)",
    "ridge": "Ridge",
    "ols": "OLS",
    "arvore_decisao": "Árvore de Decisão",
    "mlp": "MLP",
    "knn": "KNN",
    "baseline_mediana": "Baseline (mediana)",
}


def rotular(nome: str) -> str:
    return ROTULOS.get(nome, nome)


def _base(figura: go.Figure, altura: int = 420, titulo_x: str = "", titulo_y: str = "") -> go.Figure:
    """Cromo comum: grade fina e solida, eixo recessivo, sem moldura.

    Grade tracejada le como 'projecao' ou 'limiar' quando e so grade, entao
    fica solida, um tom acima da superficie.
    """
    figura.update_layout(
        height=altura,
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family=FONTE, color=TINTA_SECUNDARIA, size=13),
        margin=dict(l=8, r=16, t=48, b=48),
        hoverlabel=dict(font_family=FONTE, bgcolor=SUPERFICIE, bordercolor=EIXO),
        showlegend=False,
    )
    figura.update_xaxes(
        title_text=titulo_x,
        gridcolor=GRADE,
        griddash="solid",
        zeroline=False,
        linecolor=EIXO,
        tickfont=dict(color=TINTA_MUDA),
        title_font=dict(color=TINTA_SECUNDARIA),
    )
    figura.update_yaxes(
        title_text=titulo_y,
        gridcolor=GRADE,
        griddash="solid",
        zeroline=False,
        linecolor=EIXO,
        tickfont=dict(color=TINTA_MUDA),
        title_font=dict(color=TINTA_SECUNDARIA),
    )
    return figura


def _vazio(mensagem: str = "sem dados") -> go.Figure:
    """Aba nunca estoura traceback na cara de quem apresenta."""
    figura = go.Figure()
    figura.add_annotation(
        text=mensagem, showarrow=False, font=dict(family=FONTE, color=TINTA_MUDA, size=14)
    )
    figura.update_xaxes(visible=False)
    figura.update_yaxes(visible=False)
    return _base(figura, altura=220)


def ranking_cv(resultados: pd.DataFrame, destacar: str = VENCEDOR) -> go.Figure:
    """Barras horizontais do MAE de CV. Enfase no vencedor, resto recessivo."""
    if resultados is None or len(resultados) == 0:
        return _vazio()

    tabela = resultados.sort_values("cv_mae_log_media", ascending=False)
    cores = [AZUL if m == destacar else EIXO for m in tabela["modelo"]]

    figura = go.Figure(
        go.Bar(
            x=tabela["cv_mae_log_media"],
            y=[rotular(m) for m in tabela["modelo"]],
            orientation="h",
            marker=dict(color=cores),
            # rotulo direto so onde interessa: o valor de cada barra, fora dela,
            # que e curto e nao compete com nada
            text=[f"{v:.4f}" for v in tabela["cv_mae_log_media"]],
            textposition="outside",
            textfont=dict(color=TINTA_SECUNDARIA, size=12),
            hovertemplate="%{y}<br>MAE(log) na CV: %{x:.4f}<extra></extra>",
        )
    )
    figura.update_layout(bargap=0.35)
    return _base(figura, altura=380, titulo_x="MAE em log — validação cruzada (menor é melhor)")


def folds(por_fold: pd.DataFrame, destacar: str = VENCEDOR) -> go.Figure:
    """Um ponto por fold, uma linha por modelo.

    Escolhido em vez de seis linhas cruzando o grafico: com uma faixa por
    modelo, a afirmacao do criterio -- 'todas as cinco folds a favor' -- se le
    de imediato, porque os cinco pontos do vencedor ficam inteiramente a
    esquerda dos do segundo colocado.
    """
    if por_fold is None or len(por_fold) == 0:
        return _vazio()

    ordem = por_fold.groupby("modelo")["mae_log"].mean().sort_values(ascending=False)
    figura = go.Figure()

    for modelo in ordem.index:
        pontos = por_fold[por_fold["modelo"] == modelo]
        destaque = modelo == destacar
        figura.add_trace(
            go.Scatter(
                x=pontos["mae_log"],
                y=[rotular(modelo)] * len(pontos),
                mode="markers",
                marker=dict(
                    size=11,
                    color=AZUL if destaque else EIXO,
                    # anel da cor da superficie separa pontos sobrepostos sem
                    # desenhar borda em volta da marca
                    line=dict(width=2, color=SUPERFICIE),
                ),
                customdata=pontos["fold"],
                hovertemplate="%{y}<br>fold %{customdata}: %{x:.4f}<extra></extra>",
            )
        )
        figura.add_trace(
            go.Scatter(
                x=[pontos["mae_log"].mean()],
                y=[rotular(modelo)],
                mode="markers",
                marker=dict(size=15, symbol="line-ns", line=dict(width=2.5, color=TINTA if destaque else TINTA_MUDA)),
                hovertemplate="%{y}<br>média: %{x:.4f}<extra></extra>",
            )
        )

    return _base(figura, altura=380, titulo_x="MAE em log por fold (traço vertical = média)")


def pca_dumbbell(pca: pd.DataFrame) -> go.Figure:
    """Sem PCA -> com PCA, ligados. Duas cores porque sao dois estados, nao seis modelos."""
    if pca is None or len(pca) == 0:
        return _vazio()

    tabela = pca.sort_values("cv_mae_log_media", ascending=False)
    figura = go.Figure()

    for _, linha in tabela.iterrows():
        figura.add_trace(
            go.Scatter(
                x=[linha["cv_mae_log_media"], linha["cv_mae_log_pca_media"]],
                y=[rotular(linha["modelo"])] * 2,
                mode="lines",
                line=dict(color=EIXO, width=2),
                hoverinfo="skip",
            )
        )

    figura.add_trace(
        go.Scatter(
            x=tabela["cv_mae_log_media"],
            y=[rotular(m) for m in tabela["modelo"]],
            mode="markers",
            name="sem PCA",
            marker=dict(size=12, color=AZUL, line=dict(width=2, color=SUPERFICIE)),
            hovertemplate="%{y}<br>sem PCA: %{x:.4f}<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=tabela["cv_mae_log_pca_media"],
            y=[rotular(m) for m in tabela["modelo"]],
            mode="markers",
            name="com PCA",
            marker=dict(size=12, color=LARANJA, line=dict(width=2, color=SUPERFICIE)),
            hovertemplate="%{y}<br>com PCA: %{x:.4f}<extra></extra>",
        )
    )

    figura = _base(figura, altura=400, titulo_x="MAE em log na CV (menor é melhor)")
    # duas series: legenda sempre presente, para a identidade nao depender so da cor
    figura.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figura


def previsto_vs_real(residuos: pd.DataFrame) -> go.Figure:
    """Log-log com a diagonal do acerto perfeito. WebGL: sao 3.087 pontos."""
    if residuos is None or len(residuos) == 0:
        return _vazio()

    figura = go.Figure(
        go.Scattergl(
            x=residuos["preco_real"],
            y=residuos["preco_previsto"],
            mode="markers",
            marker=dict(size=5, color=AZUL, opacity=0.35),
            hovertemplate="real: R$ %{x:,.0f}<br>previsto: R$ %{y:,.0f}<extra></extra>",
        )
    )
    limite = [residuos["preco_real"].min(), residuos["preco_real"].max()]
    figura.add_trace(
        go.Scattergl(
            x=limite,
            y=limite,
            mode="lines",
            line=dict(color=TINTA_MUDA, width=2),
            hoverinfo="skip",
        )
    )
    figura.update_xaxes(type="log")
    figura.update_yaxes(type="log")
    return _base(
        figura, altura=460, titulo_x="preço real (R$, log)", titulo_y="preço previsto (R$, log)"
    )


def distribuicao_residuo(residuos: pd.DataFrame) -> go.Figure:
    if residuos is None or len(residuos) == 0:
        return _vazio()

    figura = go.Figure(
        go.Histogram(
            x=residuos["residuo_log"],
            nbinsx=80,
            marker=dict(color=AZUL),
            hovertemplate="resíduo: %{x:.3f}<br>anúncios: %{y}<extra></extra>",
        )
    )
    figura.add_vline(x=0, line=dict(color=TINTA_MUDA, width=2))
    return _base(
        figura, altura=360, titulo_x="resíduo em log (previsto − real)", titulo_y="anúncios"
    )


def erro_por_segmento(segmentos: pd.DataFrame, segmento: str = "bairro", topo: int = 15) -> go.Figure:
    """Erro mediano por categoria. Uma serie, uma cor -- categoria e nominal."""
    if segmentos is None or len(segmentos) == 0:
        return _vazio()

    recorte = segmentos[segmentos["segmento"] == segmento]
    if len(recorte) == 0:
        return _vazio(f"sem dados para o segmento '{segmento}'")

    recorte = recorte.nlargest(topo, "erro_mediano_percentual").sort_values(
        "erro_mediano_percentual"
    )
    figura = go.Figure(
        go.Bar(
            x=recorte["erro_mediano_percentual"],
            y=recorte["categoria"].astype(str),
            orientation="h",
            marker=dict(color=AZUL),
            customdata=recorte["n"],
            hovertemplate="%{y}<br>erro mediano: %{x:.1f}%<br>n = %{customdata}<extra></extra>",
        )
    )
    figura.update_layout(bargap=0.3)
    altura = max(320, 26 * len(recorte) + 110)
    return _base(figura, altura=altura, titulo_x="erro percentual mediano")


def importancia(tabela: pd.DataFrame, topo: int = 20) -> go.Figure:
    """Barras da importancia por permutacao.

    Cinza nao e 'outra serie': marca o que nao se distingue do proprio ruido da
    permutacao. Quem le decide o que fazer com isso -- o grafico so nao finge
    que a barra e solida.
    """
    if tabela is None or len(tabela) == 0:
        return _vazio()

    recorte = tabela.nlargest(topo, "importancia").sort_values("importancia")
    significativa = (
        recorte["significativa"]
        if "significativa" in recorte.columns
        else pd.Series([True] * len(recorte), index=recorte.index)
    )
    cores = [AZUL if s else EIXO for s in significativa]

    figura = go.Figure(
        go.Bar(
            x=recorte["importancia"],
            y=recorte["feature"],
            orientation="h",
            marker=dict(color=cores),
            error_x=dict(
                type="data",
                array=recorte["desvio"] if "desvio" in recorte.columns else None,
                color=TINTA_MUDA,
                thickness=1,
                width=3,
            ),
            hovertemplate="%{y}<br>importância: %{x:.4f}<extra></extra>",
        )
    )
    figura.update_layout(bargap=0.3)
    altura = max(360, 24 * len(recorte) + 110)
    return _base(figura, altura=altura, titulo_x="aumento no MAE(log) ao embaralhar o atributo")


def correlacao_vs_importancia(codificada: pd.DataFrame, topo: int = 400) -> go.Figure:
    """Onde os dois metodos discordam -- o achado mais interessante do projeto."""
    if codificada is None or len(codificada) == 0:
        return _vazio()
    if "spearman_abs" not in codificada.columns:
        return _vazio("o confronto com a correlação não está neste artefato")

    recorte = codificada.nlargest(min(topo, len(codificada)), "importancia")
    figura = go.Figure(
        go.Scattergl(
            x=recorte["spearman_abs"],
            y=recorte["importancia"],
            mode="markers",
            marker=dict(size=8, color=AZUL, opacity=0.6, line=dict(width=2, color=SUPERFICIE)),
            text=recorte["feature"],
            hovertemplate="%{text}<br>|spearman|: %{x:.3f}<br>importância: %{y:.4f}<extra></extra>",
        )
    )
    figura.add_hline(y=0, line=dict(color=TINTA_MUDA, width=1))
    return _base(
        figura,
        altura=440,
        titulo_x="|correlação de Spearman| com o preço",
        titulo_y="importância por permutação",
    )


def tsne(coordenadas: pd.DataFrame) -> go.Figure:
    """Faixa de preco tem ordem (Q1..Q5), entao rampa ordinal de um tom so.

    WebGL obrigatorio: sao 15.301 pontos.
    """
    if coordenadas is None or len(coordenadas) == 0:
        return _vazio()

    figura = go.Figure()
    faixas = [f for f in coordenadas["faixa_preco"].dropna().unique()]
    # ordena Q1..Q5 pelo rotulo, para a rampa acompanhar a ordem do preco
    faixas = sorted(faixas, key=lambda f: str(f))

    for cor, faixa in zip(RAMPA_ORDINAL, faixas):
        pontos = coordenadas[coordenadas["faixa_preco"] == faixa]
        figura.add_trace(
            go.Scattergl(
                x=pontos["tsne_1"],
                y=pontos["tsne_2"],
                mode="markers",
                name=str(faixa),
                marker=dict(size=4, color=cor, opacity=0.65),
                hovertemplate=f"{faixa}<extra></extra>",
            )
        )

    figura = _base(figura, altura=560, titulo_x="dimensão t-SNE 1", titulo_y="dimensão t-SNE 2")
    figura.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=""),
    )
    return figura


def faixa_da_previsao(central: float, inferior: float, superior: float) -> go.Figure:
    """Barra de intervalo: a estimativa central e a dispersao empirica do erro.

    Existe para o numero nao aparecer sozinho -- o erro mediano do modelo e
    15,6%, entao 'R$ 487.234' cravado seria precisao falsa.
    """
    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=[inferior, superior],
            y=["previsão", "previsão"],
            mode="lines",
            line=dict(color=EIXO, width=10),
            hoverinfo="skip",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=[central],
            y=["previsão"],
            mode="markers",
            marker=dict(size=18, color=AZUL, line=dict(width=2, color=SUPERFICIE)),
            hovertemplate="estimativa central: R$ %{x:,.0f}<extra></extra>",
        )
    )
    for valor, texto in ((inferior, "p10"), (superior, "p90")):
        figura.add_annotation(
            x=valor,
            y="previsão",
            text=f"{texto}<br>R$ {valor:,.0f}".replace(",", "."),
            showarrow=False,
            yshift=-38,
            font=dict(family=FONTE, color=TINTA_MUDA, size=11),
        )
    figura = _base(figura, altura=200, titulo_x="")
    figura.update_yaxes(visible=False)
    return figura
