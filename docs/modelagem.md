# Etapa 4 — Treino e Avaliação

**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB
**Módulos:** `src/imoveis_jp/models/dataset.py` e `src/imoveis_jp/models/train.py`
**Data:** 01/08/2026

---

## 1. Definição do problema

| Item | Valor |
|---|---|
| Tipo | Regressão |
| Alvo | `log(preco_venda)` |
| Observações | 15.987 anúncios com preço |
| Imóveis físicos distintos | 14.232 |
| Features | 99 |
| Semente | `42`, em `dataset.SEMENTE` |

O alvo vai para log porque a distribuição em reais tem **assimetria 5,92 e curtose
80,4**: a mediana é R$ 575 mil e o máximo R$ 19,8 milhões, então um punhado de
imóveis de altíssimo padrão domina o erro quadrático das 16 mil observações. Em
log a assimetria cai para −0,44. As métricas são reconvertidas para reais na
avaliação, porque é nelas que a resposta faz sentido.

---

## 2. Divisão treino/teste

**`GroupShuffleSplit`, 80/20, `random_state=42`, agrupado por imóvel físico.**

O agrupamento não é detalhe. A base tem o mesmo apartamento anunciado várias
vezes — 1.050 grupos somando 2.328 anúncios, o maior com 7 cópias — porque a
deduplicação entre portais usa uma chave que não pega repetições dentro do mesmo
portal. Com split aleatório simples, o mesmo imóvel cairia dos dois lados e a
métrica mediria memorização.

A assinatura do imóvel é `(preço arredondado ao milhar, área, quartos, banheiros,
garagens)`. Anúncio sem preço ou sem área não tem assinatura utilizável e vira
grupo próprio, em vez de se juntar a um grupo gigante de nulos.

Resultado: **12.777 no treino, 3.210 no teste, 0 grupos dos dois lados.** O
`train.py` verifica isso a cada execução e aborta se não for zero.

---

## 3. Pré-processamento — tudo dentro do `Pipeline`

```
ColumnTransformer
├── numéricas (8):  SimpleImputer(mediana, add_indicator) → StandardScaler
└── binárias (91):  passthrough
```

Imputação e escala **não** acontecem antes do split. Elas moram dentro do
`Pipeline`, então o `fit` é refeito em cada fold da validação cruzada e nenhuma
estatística do fold de validação entra no de treino.

Duas decisões que valem justificativa:

- **`add_indicator=True`.** Com `iptu` ausente em 80% dos anúncios, `area_total`
  em 61% e `suites`/`condominio` em 54%, o próprio silêncio do anunciante é
  informação. A indicadora preserva isso; imputar sem ela apagaria o sinal.
- **Mediana, não média.** Pelo mesmo motivo do log: as caudas são pesadas.

As binárias passam direto — já estão em 0/1, e padronizá-las só destruiria a
interpretação sem ganho.

---

## 4. Validação cruzada

**`GroupKFold(5)` sobre o conjunto de treino**, com os mesmos grupos do split. Se
fosse `KFold` simples, as duplicatas voltariam a atravessar os folds e a CV
herdaria o problema que o split resolveu.

O conjunto de teste é tocado **uma única vez**, no fim, depois de toda a seleção
de modelo.

---

## 5. Resultados

| Modelo | CV MAE (log) | Teste MAE | Erro % mediano | R² (log) |
|---|---|---|---|---|
| **Gradient Boosting** | **0,2334 ± 0,0039** | **R$ 179.489** | **17,6%** | **0,871** |
| Ridge | 0,3046 ± 0,0051 | R$ 255.779 | 23,6% | 0,769 |
| Baseline (mediana) | 0,6493 ± 0,0043 | R$ 424.273 | 42,5% | −0,005 |

O baseline existe como piso de sanidade: prever sempre a mediana. Seu R² de
−0,005 confirma que a montagem está correta — um baseline honesto tem que ficar
em torno de zero.

O gradient boosting erra **17,6% na mediana**. Para um imóvel de R$ 575 mil, isso
é uma faixa de cerca de R$ 100 mil — aceitável para triagem, insuficiente para
avaliação individual. O desvio da CV é pequeno (±0,004) frente à diferença entre
os modelos (0,07), então a vantagem sobre o Ridge é real, não ruído de partição.

A distância entre Ridge e boosting indica relações não-lineares e interações
relevantes — provavelmente entre área, bairro e padrão de acabamento.

---

## 6. Limitações

- **Sem separação temporal.** Nenhum dos dois JSONs brutos tem campo de data, então
  não há como validar em janela futura. Todo o resultado é interpolação dentro do
  mesmo instante de coleta. É a limitação mais séria para validade externa.
- **A extração via LLM não cobre o zap.** O ramo "descrições" do one-hot só
  existe para o chaves na mão. Por isso `origem_anuncio_*` fica na matriz, como
  controle: sem ele, o modelo aprende a diferença entre portais achando que é
  diferença entre imóveis.
- **`iptu` é proxy do alvo.** O IPTU é calculado sobre o valor venal, ou seja, é
  função do preço. Está disponível no anúncio, então não é vazamento temporal,
  mas infla a performance de um jeito que não se sustenta para imóvel novo sem
  IPTU lançado. Presente em apenas 20% da base.
- **Sem tuning de hiperparâmetros.** Os resultados são de configuração padrão. Um
  `GridSearchCV` dentro do `GroupKFold` deve melhorar o boosting.

---

## 7. Execução

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.features.build_features
.\.venv\Scripts\python.exe -m imoveis_jp.models.train
```

Saídas: `data/processed/resultados_modelos.csv` e `data/interim/relatorio_treino.json`,
que registra semente, proporção, folds e o diagnóstico de vazamento do split.
