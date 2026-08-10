# -*- coding: utf-8 -*-
"""Dashboard de comparacao dos modelos — Etapa 5 (issue #25).

    streamlit run app.py

Este arquivo e so apresentacao: layout, texto e cache. Toda a logica mora em
`imoveis_jp.dashboard`, que nao importa Streamlit -- e o que permite testar
carregamento, previsao e figuras sem subir servidor.

Nenhum numero e escrito a mao aqui. Tudo vem dos artefatos da rodada unica da
issue #25; se um deles faltar, a aba diz qual comando o regenera.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from imoveis_jp.dashboard import dados, graficos, modelo

st.set_page_config(
    page_title="Comparação de modelos — imóveis JP",
    page_icon="🏠",
    layout="wide",
)


# --------------------------------------------------------------------------
# cache -- unica camada que conhece o Streamlit
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def carregar(nome: str):
    return dados.carregar(nome)


@st.cache_resource(show_spinner="Ajustando o modelo vencedor no conjunto de treino...")
def modelo_ajustado():
    return modelo.ajustar()


@st.cache_data(show_spinner=False)
def opcoes_categoricas():
    return modelo.opcoes_categoricas()


@st.cache_data(show_spinner=False)
def linha_padrao():
    return modelo.linha_padrao()


@st.cache_data(show_spinner=False)
def faixa_incerteza():
    return modelo.faixa_de_incerteza()


def com_artefato(nome: str):
    """Devolve o artefato, ou mostra o comando que o gera e interrompe a aba."""
    try:
        return carregar(nome)
    except dados.ArtefatoAusente as ausente:
        st.warning(str(ausente))
        st.stop()


def reais(valor: float) -> str:
    return f"R$ {valor:,.0f}".replace(",", ".")


# --------------------------------------------------------------------------
# cabecalho
# --------------------------------------------------------------------------

st.title("Previsão de preços de imóveis em João Pessoa")
st.caption(
    "Comparação de seis modelos sob o mesmo split, mesmas folds e mesmo pré-processamento. "
    "Paradigmas de Aprendizagem de Máquina — UFPB."
)

aba_veredito, aba_folds, aba_pca, aba_residuos, aba_importancia, aba_tsne, aba_prever = st.tabs(
    [
        "O veredito",
        "Por fold",
        "PCA",
        "Resíduos",
        "Importância",
        "t-SNE",
        "Prever um imóvel",
    ]
)


# --------------------------------------------------------------------------
# 1. O veredito
# --------------------------------------------------------------------------

with aba_veredito:
    veredito = com_artefato("decisao")
    resultados = com_artefato("resultados")
    candidatos = dados.apenas_candidatos(resultados)

    st.subheader("O critério, declarado antes de qualquer resultado")
    st.markdown(
        """
Sempre existe uma métrica sob a qual um modelo específico ganha. Por isso a regra
foi escrita **antes** de rodar:

1. Vence quem tiver o menor **MAE médio de `GroupKFold(5)`** sobre `log(preço)`.
2. A vantagem sobre o segundo só é **declarada** se a diferença favorecer o mesmo
   modelo nas **cinco** folds **e** a diferença média for **≥ 0,005**.
3. Se qualquer condição falhar: **empate técnico**, desempatado por
   explicabilidade, custo de previsão e número de hiperparâmetros.
4. O teste é avaliado **depois**, uma vez, e **não participa da decisão**.

O limiar de 0,005 não é arbitrário: o desvio entre folds das melhores configurações
fica entre 0,0032 e 0,0043. Diferença menor que um desvio não se sustenta.
"""
    )

    st.subheader("Ranking por validação cruzada")
    st.plotly_chart(graficos.ranking_cv(candidatos), use_container_width=True)

    pareado = veredito["comparacao_pareada"]
    st.subheader(
        f"{graficos.rotular(pareado['vencedor'])} contra {graficos.rotular(pareado['desafiante'])}, fold a fold"
    )

    tabela_pareada = pd.DataFrame(
        {
            "fold": range(len(pareado["diffs_por_fold"])),
            "diferença (desafiante − vencedor)": pareado["diffs_por_fold"],
        }
    )
    tabela_pareada["favorece o vencedor?"] = tabela_pareada[
        "diferença (desafiante − vencedor)"
    ].map(lambda d: "sim" if d > 0 else "não")

    coluna_tabela, coluna_metricas = st.columns([2, 1])
    with coluna_tabela:
        st.dataframe(tabela_pareada, hide_index=True, use_container_width=True)
    with coluna_metricas:
        st.metric("Diferença média", f"{pareado['diferenca_media']:.4f}")
        st.metric("Limiar exigido", "0,0050")
        st.metric(
            "Folds a favor",
            f"{sum(1 for d in pareado['diffs_por_fold'] if d > 0)} de {len(pareado['diffs_por_fold'])}",
        )

    if veredito["resultado"] == "vantagem_declarada":
        st.success(
            f"**Vantagem declarada** para **{graficos.rotular(veredito['vencedor'])}** — "
            f"as duas condições valem: todas as folds a favor, e diferença média de "
            f"{pareado['diferenca_media']:.4f}, "
            f"{pareado['diferenca_media'] / 0.005:.0f}× o limiar."
        )
    else:
        st.info(
            f"**Empate técnico.** Desempate por explicabilidade, custo e número de "
            f"hiperparâmetros: **{graficos.rotular(veredito['vencedor'])}**."
        )

    st.divider()
    st.subheader("O conjunto de teste")
    st.caption(
        "A decisão acima já está fechada e não muda com o que vier daqui. "
        "O teste foi tocado uma única vez, depois, e serve para relatar — não para escolher."
    )

    if st.button("Revelar a avaliação no teste", type="primary"):
        tabela_teste = dados.ordenar_por_cv(resultados)[
            [
                "modelo",
                "cv_mae_log_media",
                "mae_reais_teste",
                "erro_percentual_mediano_teste",
                "r2_log_teste",
            ]
        ].copy()
        tabela_teste["modelo"] = tabela_teste["modelo"].map(graficos.rotular)
        st.dataframe(
            tabela_teste,
            hide_index=True,
            use_container_width=True,
            column_config={
                "modelo": "Modelo",
                "cv_mae_log_media": st.column_config.NumberColumn("CV MAE(log)", format="%.4f"),
                "mae_reais_teste": st.column_config.NumberColumn("Teste MAE", format="R$ %.0f"),
                "erro_percentual_mediano_teste": st.column_config.NumberColumn(
                    "Erro % mediano", format="%.1f%%"
                ),
                "r2_log_teste": st.column_config.NumberColumn("R² (log)", format="%.3f"),
            },
        )
        st.caption(
            "Ordenado pelo MAE da **CV**, não pelo do teste — ordenar pelo teste "
            "sugeriria que foi ele quem escolheu o vencedor."
        )


# --------------------------------------------------------------------------
# 2. Por fold
# --------------------------------------------------------------------------

with aba_folds:
    por_fold = dados.apenas_candidatos(com_artefato("folds"))

    st.subheader("Os seis candidatos nas mesmas cinco partições")
    st.markdown(
        "Guardar o score fold a fold — e não só a média — é o que torna a condição "
        "*“todas as folds a favor”* verificável. Os cinco pontos do vencedor ficam "
        "inteiramente à esquerda dos do segundo colocado."
    )
    st.plotly_chart(graficos.folds(por_fold), use_container_width=True)

    largo = por_fold.pivot(index="modelo", columns="fold", values="mae_log")
    largo.columns = [f"fold {c}" for c in largo.columns]
    largo["média"] = largo.mean(axis=1)
    largo["desvio"] = por_fold.groupby("modelo")["mae_log"].std().reindex(largo.index)
    largo.index = [graficos.rotular(m) for m in largo.index]
    st.dataframe(
        largo.sort_values("média").style.format("{:.4f}"), use_container_width=True
    )


# --------------------------------------------------------------------------
# 3. PCA
# --------------------------------------------------------------------------

with aba_pca:
    pca = com_artefato("pca")

    st.subheader("PCA como variante de pipeline")
    st.markdown(
        """
**Hipótese registrada antes de rodar: PCA piora.** PCA é uma projeção *linear*, e o
que falta aos modelos lineares é justamente a interação *não-linear* entre área e
bairro — rotacionar o espaço não cria essa interação, só reduz dimensão. Para os
modelos de árvore o argumento é outro: um componente principal (combinação linear
de dezenas de dummies de bairro) não tem limiar interpretável nem alinhado aos
cortes que a árvore faria sem PCA.
"""
    )
    st.plotly_chart(graficos.pca_dumbbell(pca), use_container_width=True)

    piorou = int(pca["piorou"].sum())
    if piorou == len(pca):
        st.success(
            f"**Hipótese confirmada nos {len(pca)} candidatos, sem exceção.** "
            f"O melhor modelo com PCA ({pca['cv_mae_log_pca_media'].min():.4f}) fica pior "
            f"que o pior modelo linear sem PCA — 50 componentes lineares desmontam o "
            f"ranking inteiro."
        )
    else:
        st.info(f"PCA piorou {piorou} de {len(pca)} candidatos.")

    exibicao = pca.copy()
    exibicao["modelo"] = exibicao["modelo"].map(graficos.rotular)
    st.dataframe(
        exibicao[
            ["modelo", "cv_mae_log_media", "cv_mae_log_pca_media", "diferenca", "n_componentes_pca"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "modelo": "Modelo",
            "cv_mae_log_media": st.column_config.NumberColumn("sem PCA", format="%.4f"),
            "cv_mae_log_pca_media": st.column_config.NumberColumn("com PCA", format="%.4f"),
            "diferenca": st.column_config.NumberColumn("piora", format="%.4f"),
            "n_componentes_pca": st.column_config.NumberColumn("componentes (95% var.)"),
        },
    )


# --------------------------------------------------------------------------
# 4. Residuos
# --------------------------------------------------------------------------

with aba_residuos:
    residuos = com_artefato("residuos")
    segmentos = com_artefato("segmentos")

    st.subheader(f"Onde o modelo erra — {len(residuos):,} anúncios do teste".replace(",", "."))

    esquerda, direita = st.columns(2)
    with esquerda:
        st.plotly_chart(graficos.previsto_vs_real(residuos), use_container_width=True)
    with direita:
        st.plotly_chart(graficos.distribuicao_residuo(residuos), use_container_width=True)

    disponiveis = segmentos["segmento"].unique().tolist()
    escolhido = st.selectbox("Erro por segmento", disponiveis, index=disponiveis.index("bairro") if "bairro" in disponiveis else 0)
    st.plotly_chart(graficos.erro_por_segmento(segmentos, escolhido), use_container_width=True)
    st.caption("Só segmentos com n ≥ 30: a mediana de quatro imóveis não sustenta conclusão.")

    st.dataframe(
        segmentos[segmentos["segmento"] == escolhido][
            ["categoria", "n", "preco_mediano", "vies_percentual", "erro_mediano_percentual"]
        ],
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# 5. Importancia
# --------------------------------------------------------------------------

with aba_importancia:
    importancia = com_artefato("importancia")
    codificada = com_artefato("importancia_codificada")

    st.subheader("Importância por permutação")
    st.caption(
        "Quanto o erro piora quando o atributo vira ruído. Medido no teste: a pergunta "
        "é do que o modelo precisa para acertar num imóvel que nunca viu. "
        "Em cinza, o que não se distingue do próprio ruído da permutação."
    )
    st.plotly_chart(graficos.importancia(importancia), use_container_width=True)

    st.subheader("Correlação × importância — onde os dois discordam")
    st.markdown(
        """
A correlação é bivariada; o modelo é multivariado. Onde as duas listas divergem está
o que só um dos métodos enxerga — e é o achado mais interessante do projeto.
`com_closet` tem correlação 0,178 com o preço e importância **exatamente zero**:
closet aparece em apartamento grande de bairro caro, então, dados `area_util` e
`bairro`, ele não acrescenta nada. A correlação estava medindo o efeito de outra
variável através dele.
"""
    )
    st.plotly_chart(graficos.correlacao_vs_importancia(codificada), use_container_width=True)


# --------------------------------------------------------------------------
# 6. t-SNE
# --------------------------------------------------------------------------

with aba_tsne:
    coordenadas = com_artefato("tsne")

    st.subheader("t-SNE 2D, colorido por faixa de preço")
    st.info(
        "Isto é **EDA, não entra em modelo nenhum**. O t-SNE não tem `.transform` em dado "
        "novo — cada chamada reprojeta o conjunto inteiro —, então não serve como passo de "
        "um Pipeline que precisa prever fora da amostra.",
        icon="ℹ️",
    )
    st.plotly_chart(graficos.tsne(coordenadas), use_container_width=True)
    st.markdown(
        "Os aglomerados pequenos e bem definidos vêm da coincidência exata de padrão "
        "categórico (sobretudo bairro). Dentro da maioria deles as faixas de preço aparecem "
        "**misturadas** — coerente com o resto do projeto: dois imóveis do mesmo bairro ainda "
        "têm preços bem diferentes se a área mudar, porque o preço depende da *combinação* "
        "área × bairro, não do bairro isolado."
    )


# --------------------------------------------------------------------------
# 7. Prever
# --------------------------------------------------------------------------

with aba_prever:
    st.subheader("Prever o preço de um imóvel")

    modo = st.radio(
        "Como quer testar?",
        ["Sortear um imóvel real do conjunto de teste", "Preencher os campos manualmente"],
        horizontal=True,
    )

    faixa = faixa_incerteza()

    if modo.startswith("Sortear"):
        st.markdown(
            f"O conjunto de teste tem **{len(com_artefato('residuos')):,} anúncios reais que o "
            f"modelo nunca viu no treino**. É o único lugar onde o imóvel é ao mesmo tempo "
            f"real e fora da amostra — por isso a demonstração é honesta.".replace(",", ".")
        )

        if "semente_sorteio" not in st.session_state:
            st.session_state.semente_sorteio = 42
        if st.button("Sortear outro imóvel", type="primary"):
            st.session_state.semente_sorteio += 1

        sorteado = modelo.sortear_do_teste(semente=st.session_state.semente_sorteio)

        a, b, c = st.columns(3)
        a.metric("O modelo previu", reais(sorteado["preco_previsto"]))
        b.metric("Preço real do anúncio", reais(sorteado["preco_real"]))
        c.metric(
            "Erro",
            reais(abs(sorteado["erro_reais"])),
            delta=f"{sorteado['erro_percentual']:+.1f}%",
            delta_color="inverse",
        )

        st.write(
            f"**{sorteado['bairro']}** · {sorteado['area_util']:.0f} m² · "
            f"{sorteado['tipo_unidade']} · faixa {sorteado['faixa_preco']} · "
            f"origem: {sorteado['origem_anuncio']}"
        )
        if sorteado["url_anuncio"]:
            st.link_button("Abrir o anúncio original", sorteado["url_anuncio"])

    else:
        opcoes = opcoes_categoricas()
        linha = linha_padrao().copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            area_util = st.number_input("Área útil (m²)", min_value=15.0, max_value=2000.0, value=65.0, step=5.0)
            area_total = st.number_input("Área total (m²)", min_value=15.0, max_value=5000.0, value=66.0, step=5.0)
            bairro = st.selectbox("Bairro", opcoes["bairro"])
        with col2:
            quartos = st.number_input("Quartos", min_value=0, max_value=10, value=2)
            suites = st.number_input("Suítes", min_value=0, max_value=5, value=1)
            banheiros = st.number_input("Banheiros", min_value=1, max_value=10, value=2)
        with col3:
            garagens = st.number_input("Garagens", min_value=0, max_value=8, value=1)
            tipo_unidade = st.selectbox("Tipo", opcoes["tipo_unidade"])
            status_construcao = st.selectbox("Status da obra", opcoes["status_construcao"])

        col4, col5 = st.columns(2)
        with col4:
            origem_anuncio = st.selectbox("Portal de origem", opcoes["origem_anuncio"])
        with col5:
            venda_direta = st.checkbox("Venda direta / leilão de banco")

        with st.expander("Comodidades (só as que a permutação mostrou importar)"):
            marcadas = {
                comodidade: st.checkbox(comodidade.replace("com_", "").replace("_", " "))
                for comodidade in modelo.COMODIDADES_UTEIS
            }

        with st.expander("Condomínio e IPTU (deixe em branco se não souber)"):
            st.caption(
                "Em branco vai como **nulo**, não como a mediana. O pipeline treinou com "
                "`add_indicator=True`: a ausência é sinal que o modelo aprendeu a usar — "
                "o IPTU falta em cerca de 80% dos anúncios."
            )
            usar_condominio = st.checkbox("Informar condomínio")
            condominio = st.number_input("Condomínio (R$)", min_value=0.0, value=280.0, step=50.0) if usar_condominio else None
            usar_iptu = st.checkbox("Informar IPTU")
            iptu = st.number_input("IPTU (R$)", min_value=0.0, value=300.0, step=50.0) if usar_iptu else None

        linha.loc[0, "area_util"] = area_util
        linha.loc[0, "area_total"] = area_total
        linha.loc[0, "bairro"] = bairro
        linha.loc[0, "quartos"] = float(quartos)
        linha.loc[0, "suites"] = float(suites)
        linha.loc[0, "banheiros"] = float(banheiros)
        linha.loc[0, "garagens"] = float(garagens)
        linha.loc[0, "tipo_unidade"] = tipo_unidade
        linha.loc[0, "status_construcao"] = status_construcao
        linha.loc[0, "origem_anuncio"] = origem_anuncio
        if "venda_direta" in linha.columns:
            linha.loc[0, "venda_direta"] = int(venda_direta)
        for comodidade, marcada in marcadas.items():
            if comodidade in linha.columns:
                linha.loc[0, comodidade] = int(marcada)
        if condominio is not None:
            linha.loc[0, "condominio"] = condominio
        if iptu is not None:
            linha.loc[0, "iptu"] = iptu

        previsao = modelo.prever(modelo_ajustado(), linha)

        st.divider()
        st.metric("Estimativa central", reais(previsao.central))
        st.plotly_chart(
            graficos.faixa_da_previsao(previsao.central, previsao.inferior, previsao.superior),
            use_container_width=True,
        )
        st.caption(
            f"A faixa não é intervalo de confiança estatístico: é a dispersão empírica do erro "
            f"que o modelo cometeu no teste (quantis {faixa['inferior']:.1f}% e "
            f"+{faixa['superior']:.1f}%), aplicada a esta previsão. O erro mediano do modelo é "
            f"{faixa['mediano_absoluto']:.1f}% — mostrar só o número central seria precisão falsa."
        )

        for aviso in previsao.avisos:
            st.warning(aviso, icon="⚠️")
