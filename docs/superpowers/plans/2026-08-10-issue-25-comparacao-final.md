# Issue #25 — Comparação Final, Critério de Decisão e Única Avaliação no Teste — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Nota desta sessão:** este plano foi escrito e executado inline, na mesma sessão, pelo mesmo agente que já tinha todo o contexto do repositório carregado — não houve handoff para subagente "zero-contexto". Os arquivos e o nível de detalhe abaixo continuam válidos para retomar ou auditar o trabalho depois.

**Goal:** Fechar a Fase 2 (issue #25): comparar os seis candidatos registrados sob o mesmo split/folds, guardar o score fold a fold, aplicar o critério de decisão declarado *antes* de tocar no teste, medir a variante PCA, rodar t-SNE como EDA, consolidar a documentação e regenerar os artefatos processados uma única vez.

**Architecture:** Reaproveita a infraestrutura existente (`dataset.py`, `train.py`, `candidatos/*.py`) sem alterar o contrato do `Candidato`. Adiciona: (1) captura do score por fold dentro do `train.py`; (2) um módulo novo e puro (`decisao.py`) que só lê os scores por fold e decide, nunca toca no teste; (3) um módulo de variante PCA que reusa `train.montar_preprocessador`; (4) um módulo de EDA (t-SNE) isolado em `features/`, fora do fluxo de modelagem; (5) documentação markdown consolidando tudo.

**Tech Stack:** Python 3.10, scikit-learn 1.7.2 (`.venv` já tem tudo — `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`), pandas, matplotlib. Sem dependência nova.

## Global Constraints

- `dataset.SEMENTE = 42` em qualquer estimador estocástico (regra já testada em `test_candidatos.py`).
- Mesmo split (`dataset.dividir`), mesmo `GroupKFold(5)`, mesmo `Pipeline` de pré-processamento para todos os modelos comparados — regra 1 do protocolo.
- O conjunto de teste só é tocado pelo `train.py::executar()`, que já roda uma única vez por invocação. `decisao.py` não deve importar nem ler nenhuma coluna `*_teste`.
- Limiar de vantagem declarável: `>= 0.005` de diferença média, **e** a mesma direção nas 5 folds (issue #25, seção "Regra").
- Os seis candidatos que entram na decisão: `arvore_decisao`, `knn`, `mlp`, `ols`, `ridge`, `gradient_boosting_ajustado`. `baseline_mediana` e `gradient_boosting` (padrão) são referências, não candidatos (protocolo §3.1) — aparecem nas tabelas de contexto mas não competem pela decisão.
- `data/processed/*.csv` e `docs/figuras/*.png` são regenerados **uma única vez**, no final, depois que todo o código estiver pronto — para não gerar múltiplas rodadas conflitantes (entregável explícito da issue).
- Todo texto novo em `docs/` segue o português do resto do projeto e não inventa número — todo valor citado vem de um CSV/JSON gerado pelos scripts desta issue.

---

## File Structure

- Modify: `src/imoveis_jp/models/train.py` — grava o score por fold além da média.
- Create: `src/imoveis_jp/models/decisao.py` — critério de decisão (regras 1–3 do protocolo).
- Create: `tests/test_decisao.py` — testa o critério com dados sintéticos.
- Create: `src/imoveis_jp/models/pca_variant.py` — variante PCA sobre os seis candidatos.
- Create: `src/imoveis_jp/features/tsne_exploracao.py` — t-SNE 2D, EDA, colorido por faixa de preço.
- Create: `docs/modelos/ridge.md`, `docs/modelos/ols.md`, `docs/modelos/knn.md`, `docs/modelos/gradient_boosting.md` — docs individuais que faltavam (issues #22/#24 fecharam sem entregar o markdown; ridge/gradient_boosting nunca tiveram issue própria). Template curto definido na própria issue #25.
- Create: `docs/comparacao_modelos.md` — o entregável principal, consolidando tudo.
- Regenerate (last step, once): `data/processed/resultados_modelos.csv`, `data/processed/cv_mae_por_fold.csv`, `data/processed/decisao_criterio.json`, `data/processed/resultados_pca.csv`, `data/processed/tsne_coords.csv`, `docs/figuras/tsne_precos.png`, `data/interim/relatorio_treino.json`.

---

### Task 1: Capturar o score por fold em `train.py`

**Files:**
- Modify: `src/imoveis_jp/models/train.py:36-37` (constantes de saída), `:203-276` (`executar`)

**Interfaces:**
- Produces: `data/processed/cv_mae_por_fold.csv` com colunas `modelo, fold, mae_log` — uma linha por (modelo, fold). Consumido pelo Task 2.

- [ ] **Step 1: Adicionar a constante de saída**

```python
# logo abaixo de SAIDA_RELATORIO em train.py
SAIDA_CV_FOLDS = config.PROCESSED / "cv_mae_por_fold.csv"
```

- [ ] **Step 2: Acumular o score por fold dentro do laço de `executar()`**

Em `executar()`, antes do laço `for indice, (nome, modelo) in enumerate(...)`, adicionar:

```python
    linhas = []
    linhas_por_fold = []
```

(a lista `linhas` já existe; só adicionar `linhas_por_fold` ao lado dela).

Logo depois do bloco que calcula `mae_cv = -scores` (já existe, não mexer no cálculo), adicionar:

```python
        for indice_fold, valor in enumerate(mae_cv):
            linhas_por_fold.append({"modelo": nome, "fold": indice_fold, "mae_log": float(valor)})
```

`GroupKFold` não embaralha — a atribuição de fold é determinística a partir de `g_tr`, que é idêntico para todo modelo desta execução. Por isso o índice de fold é comparável entre modelos sem precisar recriar o split manualmente.

- [ ] **Step 3: Persistir o CSV junto dos outros artefatos**

Logo após a linha `resultados.to_csv(SAIDA_RESULTADOS, ...)`, adicionar:

```python
    cv_por_fold = pd.DataFrame(linhas_por_fold)
    cv_por_fold.to_csv(SAIDA_CV_FOLDS, index=False, encoding="utf-8")
```

E no bloco de log final, junto de `_log(f"Resultados: {SAIDA_RESULTADOS}")`, adicionar:

```python
    _log(f"CV por fold: {SAIDA_CV_FOLDS}")
```

- [ ] **Step 4: Rodar a suíte existente para garantir que nada quebrou**

Run: `.venv/bin/python -m pytest tests/test_candidatos.py tests/test_models.py -q`
Expected: PASS (o teste não chama `train.executar()`, só `montar_modelos`/`montar_preprocessador`, então este diff não deveria mudar nenhum resultado de teste).

- [ ] **Step 5: Commit** (opcional — ver nota de execução no fim do plano; commits só acontecem se o usuário pedir)

---

### Task 2: `decisao.py` — o critério de decisão como código

**Files:**
- Create: `src/imoveis_jp/models/decisao.py`
- Test: `tests/test_decisao.py`

**Interfaces:**
- Consumes: `data/processed/cv_mae_por_fold.csv` (produzido pelo Task 1), `candidatos.descobrir()` (já existe, devolve `Dict[str, Candidato]`).
- Produces: função pública `decidir(fold_scores: pd.DataFrame, inscritos: Dict[str, Candidato]) -> dict` e `executar() -> dict`. O dict de `decidir` tem as chaves `ranking_cv`, `medias_cv`, `comparacao_pareada`, `resultado` (`"vantagem_declarada"` ou `"empate_tecnico"`), `vencedor`.

- [ ] **Step 1: Escrever o teste com dados sintéticos (falha primeiro)**

```python
# tests/test_decisao.py
import pandas as pd
import pytest

from imoveis_jp.models import decisao
from imoveis_jp.models.candidatos.base import Candidato
from sklearn.linear_model import LinearRegression


def _candidato(nome, grade=None):
    return Candidato(
        nome=nome,
        dono="teste",
        regressor=LinearRegression(),
        hipotese="hipotese sintetica so para o teste do criterio de decisao passar",
        grade=grade or {},
    )


def _folds(valores_por_modelo):
    """valores_por_modelo: {'a': [f0..f4], 'b': [f0..f4]}"""
    linhas = [
        {"modelo": nome, "fold": i, "mae_log": v}
        for nome, valores in valores_por_modelo.items()
        for i, v in enumerate(valores)
    ]
    return pd.DataFrame(linhas)


def test_vantagem_declarada_quando_todas_as_folds_concordam_e_margem_e_grande():
    fold_scores = _folds({
        "a": [0.20, 0.21, 0.19, 0.20, 0.205],
        "b": [0.22, 0.23, 0.21, 0.22, 0.225],
    })
    inscritos = {"a": _candidato("a"), "b": _candidato("b")}

    veredito = decisao.decidir(fold_scores, inscritos)

    assert veredito["resultado"] == "vantagem_declarada"
    assert veredito["vencedor"] == "a"
    assert veredito["comparacao_pareada"]["diferenca_media"] >= decisao.LIMIAR_VANTAGEM


def test_empate_tecnico_quando_uma_fold_discorda():
    fold_scores = _folds({
        "a": [0.20, 0.21, 0.19, 0.20, 0.230],  # ultima fold pior que 'b'
        "b": [0.22, 0.23, 0.21, 0.22, 0.225],
    })
    inscritos = {"a": _candidato("a"), "b": _candidato("b")}

    veredito = decisao.decidir(fold_scores, inscritos)

    assert veredito["resultado"] == "empate_tecnico"
    assert veredito["comparacao_pareada"]["todas_as_folds_a_favor"] is False


def test_empate_tecnico_quando_margem_fica_abaixo_do_limiar():
    fold_scores = _folds({
        "a": [0.2000, 0.2100, 0.1900, 0.2000, 0.2050],
        "b": [0.2020, 0.2120, 0.1920, 0.2020, 0.2070],  # diferenca de 0.002 < 0.005
    })
    inscritos = {"a": _candidato("a"), "b": _candidato("b")}

    veredito = decisao.decidir(fold_scores, inscritos)

    assert veredito["resultado"] == "empate_tecnico"
    assert veredito["comparacao_pareada"]["todas_as_folds_a_favor"] is True
    assert veredito["comparacao_pareada"]["diferenca_media"] < decisao.LIMIAR_VANTAGEM


def test_desempate_usa_explicabilidade_custo_e_numero_de_hiperparametros():
    empatados = ["mlp", "ols"]
    inscritos = {
        "mlp": _candidato("mlp", grade={"regressor__alpha": [1]}),
        "ols": _candidato("ols", grade={}),
    }
    ordem = decisao.desempate(empatados, inscritos)
    # ols e mais explicavel e mais barato que mlp nas tabelas hardcoded do modulo
    assert ordem[0] == "ols"


def test_recusa_candidato_fora_do_registro():
    fold_scores = _folds({"fantasma": [0.2] * 5})
    with pytest.raises(KeyError):
        decisao.decidir(fold_scores, inscritos={})
```

- [ ] **Step 2: Rodar o teste para confirmar que falha**

Run: `.venv/bin/python -m pytest tests/test_decisao.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'imoveis_jp.models.decisao'`

- [ ] **Step 3: Escrever `decisao.py`**

```python
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
```

- [ ] **Step 4: Rodar o teste de novo**

Run: `.venv/bin/python -m pytest tests/test_decisao.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Commit**

---

### Task 3: Variante PCA sobre os seis candidatos

**Files:**
- Create: `src/imoveis_jp/models/pca_variant.py`

**Interfaces:**
- Consumes: `train.montar_preprocessador`, `train.SAIDA_RESULTADOS` (para a linha "sem PCA" já calculada no Task 1/5), `decisao.CANDIDATOS_NA_DECISAO`.
- Produces: `data/processed/resultados_pca.csv` com colunas `modelo, cv_mae_log_pca_media, cv_mae_log_pca_desvio, n_componentes_pca, cv_mae_log_media, diferenca, piorou`.

- [ ] **Step 1: Escrever o módulo**

```python
# -*- coding: utf-8 -*-
"""Variante de pipeline com PCA, aplicada aos seis candidatos (issue #25).

Hipotese registrada ANTES de rodar: PCA piora. PCA e uma projecao LINEAR, e o
que falta para os modelos lineares (Ridge/OLS) e justamente a interacao
NAO-linear entre area e bairro -- uma rotacao linear do espaco de atributos nao
cria essa interacao, so reduz dimensao. Para os modelos baseados em arvore o
argumento e outro: eles cortam por limiar em atributos individuais, e um
componente principal (combinacao linear de dezenas de dummies de bairro) nao
tem um limiar interpretavel nem alinhado aos cortes que a arvore faria sem PCA.
E tambem destroi a interpretabilidade que da o resultado mais forte do projeto
(bairro isolado e o atributo mais importante, ver docs/modelagem.md).

Demonstrar que piora, com numero, vale mais que omitir.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from imoveis_jp import config
from imoveis_jp.models import candidatos, dataset, train
from imoveis_jp.models.decisao import CANDIDATOS_NA_DECISAO

SAIDA = config.PROCESSED / "resultados_pca.csv"

#: variancia retida -- escolha reprodutivel e unica para os seis candidatos,
#: em vez de um numero fixo de componentes que faria sentido para um modelo e
#: nao para outro.
VARIANCIA_RETIDA = 0.95


def _log(msg: str) -> None:
    print(msg, flush=True)


def montar_pipeline_pca(candidato, preprocessador) -> Pipeline:
    return Pipeline(
        [
            ("preparo", preprocessador),
            ("pca", PCA(n_components=VARIANCIA_RETIDA, random_state=dataset.SEMENTE)),
            ("regressor", candidato.regressor),
        ]
    )


def executar() -> pd.DataFrame:
    config.ensure_dirs()

    if not train.SAIDA_RESULTADOS.exists():
        raise FileNotFoundError(
            f"'{train.SAIDA_RESULTADOS}' nao existe. Rode antes: python -m imoveis_jp.models.train"
        )

    X, y, grupos = dataset.carregar()
    X_tr, X_te, y_tr, y_te, g_tr, g_te = dataset.dividir(X, y, grupos)
    numericas, binarias, categoricas = dataset.colunas_por_tipo(X)
    inscritos = candidatos.descobrir()

    linhas = []
    for indice, nome in enumerate(CANDIDATOS_NA_DECISAO, start=1):
        candidato = inscritos[nome]
        preprocessador = train.montar_preprocessador(
            numericas, binarias, categoricas, candidato.escalar_binarias
        )
        modelo = montar_pipeline_pca(candidato, preprocessador)

        _log(f"[PCA {indice}/{len(CANDIDATOS_NA_DECISAO)}] {nome}...")
        scores = cross_val_score(
            modelo,
            X_tr,
            y_tr,
            groups=g_tr,
            cv=GroupKFold(n_splits=train.FOLDS),
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )
        mae_pca = -scores

        modelo.fit(X_tr, y_tr)
        n_componentes = int(modelo.named_steps["pca"].n_components_)

        _log(
            f"[PCA] {nome}: MAE(log)={mae_pca.mean():.4f} +/- {mae_pca.std():.4f} "
            f"({n_componentes} componentes p/ {VARIANCIA_RETIDA:.0%} da variancia)"
        )
        linhas.append(
            {
                "modelo": nome,
                "cv_mae_log_pca_media": float(mae_pca.mean()),
                "cv_mae_log_pca_desvio": float(mae_pca.std()),
                "n_componentes_pca": n_componentes,
            }
        )

    resultado = pd.DataFrame(linhas)
    sem_pca = pd.read_csv(train.SAIDA_RESULTADOS)[["modelo", "cv_mae_log_media"]]
    comparacao = resultado.merge(sem_pca, on="modelo", how="left")
    comparacao["diferenca"] = comparacao["cv_mae_log_pca_media"] - comparacao["cv_mae_log_media"]
    comparacao["piorou"] = comparacao["diferenca"] > 0
    comparacao = comparacao.sort_values("cv_mae_log_pca_media").reset_index(drop=True)
    comparacao.to_csv(SAIDA, index=False, encoding="utf-8")

    _log("\n" + "=" * 72)
    _log("PCA -- comparacao com o pipeline sem PCA")
    _log("=" * 72)
    _log(comparacao.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    _log(f"\nSalvo em: {SAIDA}")

    return comparacao


if __name__ == "__main__":
    executar()
```

- [ ] **Step 2: Rodar manualmente depois que o Task 1/5 já tiver gerado `resultados_modelos.csv`**

Run: `.venv/bin/python -m imoveis_jp.models.pca_variant`
Expected: imprime a tabela comparativa e grava `data/processed/resultados_pca.csv`; sem exceção.

- [ ] **Step 3: Commit**

---

### Task 4: t-SNE 2D como EDA (não entra em modelo)

**Files:**
- Create: `src/imoveis_jp/features/tsne_exploracao.py`

**Interfaces:**
- Consumes: `dataset.carregar()`, `dataset.colunas_por_tipo`, `train.montar_preprocessador`.
- Produces: `data/processed/tsne_coords.csv` (colunas `tsne_1, tsne_2, faixa_preco`), `docs/figuras/tsne_precos.png`.

- [ ] **Step 1: Escrever o módulo**

```python
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
```

- [ ] **Step 2: Rodar e observar o tempo (baseline: ~15.476 linhas)**

Run: `.venv/bin/python -m imoveis_jp.features.tsne_exploracao`
Expected: termina sem erro; se demorar mais que ~5 minutos, considerar `perplexity` menor ou amostragem — decidir na hora olhando o tempo real, não travar o plano nisso.

- [ ] **Step 3: Commit**

---

### Task 5: Rodar o pipeline completo uma vez e regenerar os artefatos

**Files:** nenhum arquivo novo — só execução, na ordem que respeita as dependências dos tasks anteriores.

- [ ] **Step 1:** `.venv/bin/python -m pytest -q` (suíte inteira, garante que Tasks 1–2 não quebraram nada)
- [ ] **Step 2:** `.venv/bin/python -m imoveis_jp.models.train` (gera `resultados_modelos.csv` e `cv_mae_por_fold.csv` consolidados, com os 6 candidatos + 2 referências)
- [ ] **Step 3:** `.venv/bin/python -m imoveis_jp.models.decisao` (lê o CSV do passo 2, decide, grava `decisao_criterio.json`) — **anunciar o resultado aqui, antes de qualquer leitura de `*_teste`**
- [ ] **Step 4:** `.venv/bin/python -m imoveis_jp.models.pca_variant`
- [ ] **Step 5:** `.venv/bin/python -m imoveis_jp.features.tsne_exploracao`
- [ ] **Step 6:** `.venv/bin/python -m imoveis_jp.models.analysis` (já existia; regenera `residuos_diagnostico.png`/`importancia_permutacao.png` sobre o modelo vencedor, para as figuras ficarem consistentes com a rodada única)

---

### Task 6: Documentação individual que faltava

**Files:**
- Create: `docs/modelos/ridge.md`, `docs/modelos/ols.md`, `docs/modelos/knn.md`, `docs/modelos/gradient_boosting.md`

Cada um segue o template da própria issue #25 ("Template das docs individuais"):

```markdown
## <Modelo>
**Dono:** · **Branch:** · **PR:** #

### O que este modelo assume sobre os dados
### Hipótese registrada ANTES de rodar
### Grade testada, e por que essa faixa
### Resultado
### Por que esse resultado
### A hipótese se confirmou?
### Limitação conhecida
```

- [ ] **Step 1:** Para cada um dos quatro modelos, preencher o template puxando: a hipótese literal de `candidatos/<nome>.py` (já escrita, não reescrever), a grade de `GRADE` no mesmo arquivo, e os números reais de `data/processed/resultados_modelos.csv` + `cv_mae_por_fold.csv` gerados no Task 5. Nenhum número inventado — só o que os CSVs mostrarem.
- [ ] **Step 2:** No topo de cada arquivo, uma nota de uma linha: "Documentação escrita durante a consolidação da Fase 2 (issue #25) porque não havia sido entregue com o candidato." — transparência sobre quem escreveu e quando, sem fingir autoria retroativa.
- [ ] **Step 3:** Commit

---

### Task 7: `docs/comparacao_modelos.md` — o entregável principal

**Files:**
- Create: `docs/comparacao_modelos.md`

Seções, cada uma alimentada por um artefato específico gerado nos tasks anteriores — nenhum número escrito à mão:

1. **Escopo e critério** — recapitula as 4 regras da issue #25 e o porquê do limiar 0,005 (copiar/condensar da issue, já com a fonte).
2. **Comparação pareada por fold** — tabela modelo × fold0..fold4 × média × desvio, de `cv_mae_por_fold.csv`, para os 6 candidatos.
3. **Aplicação do critério** — o conteúdo de `decisao_criterio.json` em prosa: ranking, comparação pareada vencedor×segundo, resultado (`vantagem_declarada` ou `empate_tecnico`), desempate se houve.
4. **Avaliação única no teste** — tabela final de `resultados_modelos.csv` (colunas `*_teste`), para os 8 modelos (6 candidatos + 2 referências), ordenada pelo MAE de CV — não pelo MAE de teste, para não parecer que o teste escolheu.
5. **Variante PCA** — tabela de `resultados_pca.csv`, com a hipótese registrada em `pca_variant.py` e se confirmou ou não, explicando o mecanismo (linear vs. interação não-linear; perda de interpretabilidade do bairro).
6. **t-SNE (EDA)** — embutir `docs/figuras/tsne_precos.png`, 2–3 frases sobre o que o gráfico mostra (ou não mostra) sobre separação por faixa de preço, deixando explícito que isso não influenciou nenhum modelo.
7. **Docs individuais consolidadas** — link para os 6 `docs/modelos/*.md` (arvore, mlp existentes + ridge/ols/knn/gradient_boosting do Task 6), um parágrafo de resumo (hipótese + resultado + por quê) para cada.
8. **Viés em aberto** — repetir o aviso do protocolo (§4) e da issue sobre `bairro × area_util` ainda não estar explícito na matriz; registrar como limitação conhecida para uma etapa futura, não como algo resolvido aqui (fora do escopo desta issue).
9. **Conclusão** — o parágrafo final com o veredito e o porquê, sem reabrir a discussão.

- [ ] **Step 1:** Escrever o arquivo depois que os Tasks 1–6 já tiverem rodado e todos os CSVs/JSON/PNG existirem — sem isso não há números para citar.
- [ ] **Step 2:** Revisar contra o "Self-Review" abaixo.
- [ ] **Step 3:** Commit

---

## Self-Review

**1. Cobertura da issue:**
- Comparação pareada por fold → Task 1 + 2 + seção 2 do doc. ✅
- Anunciar vencedora antes do teste → Task 5 (ordem dos passos: `decisao.py` roda antes de qualquer leitura de `*_teste`) + seção 3 do doc. ✅
- Teste rodado uma vez, com todos os modelos → `train.py::executar()` já faz isso numa única chamada; Task 5 Step 2 é a única invocação. ✅
- PCA em todos → Task 3, sobre os 6 candidatos (baseline/GB-padrão excluídos por não serem candidatos — decisão documentada no próprio módulo). ✅
- t-SNE 2D, EDA, não entra em modelo → Task 4, módulo isolado em `features/`, comentário explícito no docstring sobre por que não pode virar passo de `Pipeline`. ✅
- `docs/comparacao_modelos.md` consolidando as docs individuais → Task 6 (preenche as que faltavam) + Task 7. ✅
- Regenerar artefatos uma única vez, evitando conflito de merge → Task 5, sequência única e ordenada. ✅

**2. Placeholders:** nenhum "TBD" ou "implementar depois" nos tasks de código (1–4); os tasks de doc (6–7) dependem de números que só existem após a execução — isso é sequenciamento, não placeholder, e o plano é explícito sobre qual artefato alimenta qual seção.

**3. Consistência de tipos/nomes:** `CANDIDATOS_NA_DECISAO` é definida uma vez em `decisao.py` e importada por `pca_variant.py` (Task 3) — não duplicada. `SAIDA_RESULTADOS`/`SAIDA_CV_FOLDS`/`FOLDS` reusam as constantes já existentes em `train.py` em vez de redeclarar valores.

---

## Execução

Este plano foi executado **inline, nesta mesma sessão**, pelo agente que já tinha todo o contexto do repositório carregado (dataset.py, train.py, candidatos/*, protocolo_comparacao.md, modelagem.md, issues #21-24 fechadas, docs existentes). Não houve handoff para subagente com contexto zerado, então a rodada de "Subagent-Driven vs. Inline Execution" da skill foi resolvida a favor de Inline — reabrir esse contexto em um subagente novo custaria mais do que continuar.
