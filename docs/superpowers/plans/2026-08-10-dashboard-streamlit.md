# Dashboard Streamlit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Nota:** executado inline nesta sessão, pelo agente que escreveu o spec e tem o repositório em contexto.

**Goal:** App Streamlit com 7 abas sobre os artefatos da issue #25, incluindo previsão de imóvel (manual ou sorteado do teste).

**Spec:** [`docs/superpowers/specs/2026-08-10-dashboard-streamlit-design.md`](../specs/2026-08-10-dashboard-streamlit-design.md)

**Tech Stack:** Streamlit, Plotly, pandas, scikit-learn 1.7.2 (já instalado).

## Global Constraints

- `dados.py`, `modelo.py`, `graficos.py` **não importam Streamlit**. Cache só em `app.py`.
- Todo caminho vem de `imoveis_jp.config` (regra nº 1 do README).
- Textos em PT-BR, seguindo o tom do resto do projeto.
- Nenhum número escrito à mão nos textos: tudo lido dos artefatos.
- Artefato ausente → mensagem com o comando que o regenera, não traceback.

---

## File Structure

- Create: `src/imoveis_jp/dashboard/__init__.py`
- Create: `src/imoveis_jp/dashboard/dados.py` — carregamento e validação
- Create: `src/imoveis_jp/dashboard/modelo.py` — ajuste, previsão, incerteza
- Create: `src/imoveis_jp/dashboard/graficos.py` — figuras Plotly puras
- Create: `app.py` — entry point, layout, cache
- Create: `tests/test_dashboard.py`
- Modify: `requirements.txt` — `streamlit`, `plotly`
- Modify: `README.md` — como rodar o app

---

### Task 1: Dependências e `dados.py`

**Interfaces produzidas:**
- `ARTEFATOS: Dict[str, Artefato]` — registro nome → (caminho, comando que regenera)
- `carregar(nome: str) -> pd.DataFrame | dict` — levanta `ArtefatoAusente` com o comando
- `ArtefatoAusente(FileNotFoundError)` — expõe `.comando`
- `CANDIDATOS`, `REFERENCIAS` — reexportados de `decisao.py`, sem duplicar a lista

- [ ] Adicionar `streamlit>=1.40` e `plotly>=5.24` ao `requirements.txt`, instalar no `.venv`
- [ ] Escrever `tests/test_dashboard.py::test_carrega_cada_artefato` e `::test_artefato_ausente_diz_o_comando` (falham)
- [ ] Implementar `dados.py`
- [ ] Rodar os testes: passam

### Task 2: `modelo.py`

**Interfaces produzidas:**
- `ajustar() -> Pipeline` — reajusta `gradient_boosting_ajustado` no treino
- `linha_padrao() -> pd.DataFrame` — 1 linha com todas as 76 colunas nos defaults do spec
- `prever(modelo, linha) -> Previsao` — dataclass com `central`, `inferior`, `superior`, `avisos`
- `FAIXA_INCERTEZA` — quantis 10/90 lidos de `residuos_teste.csv`, não hardcoded
- `sortear_do_teste(seed) -> dict` — imóvel real com previsão, preço real, erro, URL

- [ ] Testes: previsão finita; nulo em `condominio`/`iptu` não quebra; faixa monótona; avisos de extrapolação disparam nas bordas (área > 1.260, previsão > R$ 2M)
- [ ] Implementar `modelo.py`
- [ ] Rodar os testes

### Task 3: `graficos.py`

**Carregar a skill `dataviz` antes de escrever a primeira linha de código de gráfico.**

**Interfaces produzidas** (todas devolvem `plotly.graph_objects.Figure`):
`ranking_cv`, `folds`, `pca_dumbbell`, `previsto_vs_real`, `distribuicao_residuo`,
`erro_por_segmento`, `importancia`, `correlacao_vs_importancia`, `tsne`

- [ ] Testes: cada função devolve `Figure` e sobrevive a DataFrame vazio
- [ ] Implementar `graficos.py`
- [ ] Rodar os testes

### Task 4: `app.py`

- [ ] 7 abas conforme o spec; teste da aba 1 atrás de `st.button`; aba 7 com os dois modos
- [ ] Cache: `st.cache_data` nos carregamentos, `st.cache_resource` no ajuste do modelo
- [ ] Tratar `ArtefatoAusente` por aba, mostrando o comando

### Task 5: Verificação

- [ ] `pytest -q` completo verde
- [ ] Subir o app, navegar as 7 abas, conferir que nenhuma quebra
- [ ] Atualizar o README com o comando de execução
- [ ] Commit
