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
| Imóveis físicos distintos | 14.224 |
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

Resultado: **12.820 no treino, 3.167 no teste, 0 grupos dos dois lados.** O
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
  em 61% e `suites`/`condominio` perto de 47% e 54%, o próprio silêncio do anunciante é
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
| **Gradient Boosting ajustado** | **0,2232 ± 0,0031** | **R$ 166.660** | **16,4%** | **0,863** |
| Gradient Boosting (padrão) | 0,2306 ± 0,0034 | R$ 173.672 | 17,7% | 0,857 |
| Ridge | 0,3037 ± 0,0050 | R$ 301.659 | 23,6% | 0,744 |
| Baseline (mediana) | 0,6531 ± 0,0047 | R$ 417.354 | 42,5% | −0,001 |

O baseline existe como piso de sanidade: prever sempre a mediana. Seu R² de
−0,001 confirma que a montagem está correta — um baseline honesto tem que ficar
em torno de zero.

O gradient boosting ajustado erra **16,4% na mediana**. Para um imóvel de R$ 575 mil, isso
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
- **Folds em série, não em paralelo.** O `HistGradientBoosting` já é multi-thread
  por OpenMP; paralelizar as folds por cima sobrecarregava a máquina e um worker
  morria, devolvendo `nan` silenciosamente como score. O `train.py` agora usa
  `error_score="raise"` e aborta se algum score não for finito.
- **O early stopping usa uma validação interna não agrupada.** O
  `HistGradientBoostingRegressor` liga `early_stopping='auto'` sozinho acima de
  10.000 amostras — temos 12.820 — e separa 10% para decidir quando parar. Essa
  fatia é sorteada **aleatoriamente**, não pela assinatura do imóvel, então pode
  conter cópia de uma linha de treino e fazer o modelo parar um pouco tarde. O
  efeito é limitado (decide só o momento de parada, não seleção de atributos),
  mas é a única parte do pipeline que o agrupamento do split não cobre.

---

## 7. Execução

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.features.build_features
.\.venv\Scripts\python.exe -m imoveis_jp.models.train
```

Saídas: `data/processed/resultados_modelos.csv` e `data/interim/relatorio_treino.json`,
que registra semente, proporção, folds e o diagnóstico de vazamento do split.

---

## 8. Busca de hiperparâmetros

`src/imoveis_jp/models/tune.py`, `GridSearchCV` com o **mesmo `GroupKFold`** do
treino e sobre o **mesmo conjunto de treino**. O teste continua intocado.

### 8.1 Por que o score da busca não é o número que se relata

Escolher a melhor de 32 configurações pelo score da validação cruzada torna esse
score otimista: parte da vantagem da vencedora é sorte de partição, não
qualidade. A forma rigorosa de medir isso é validação cruzada aninhada, com um
laço externo que só mede. Não foi usada aqui porque custaria 5× o tempo para
responder uma pergunta que o conjunto de teste já responde — ele foi separado
antes da busca e não participou de nenhuma decisão.

**O score da busca serve para escolher; o score do teste serve para relatar.**

### 8.2 A primeira grade estava mal-posta

A passada inicial (48 configurações) devolveu `max_leaf_nodes=63` e
`max_iter=500` — os dois no **topo** da grade. Olhando o MAE médio por valor:

```
max_leaf_nodes   15 → 0,2346    31 → 0,2291    63 → 0,2262
max_iter        200 → 0,2313   500 → 0,2286
```

`max_leaf_nodes` melhorava monotonicamente até a borda, sinal clássico de que o
ótimo estava fora da grade. Aceitar aquele "melhor" seria reportar um ótimo de
grade, não um ótimo real. A faixa foi estendida até 255, e aí a curva virou:

```
31 → 0,2284    63 → 0,2260    127 → 0,2261    255 → 0,2271
```

Com o ótimo interior, a busca passa a estar bem-posta.

`max_iter` era um falso eixo: as duas melhores configurações com 200 e com 500
ficaram a **0,00002** uma da outra, porque o early stopping automático (§6) para
o modelo antes do teto. Virou valor fixo, documentado como limite superior — sem
isso, quem revisitasse a grade perderia tempo mexendo nele.

### 8.3 Resultado

| Modelo | CV padrão | CV ajustado | Ganho |
|---|---|---|---|
| Gradient Boosting | 0,2279 | **0,2232** | +2,1% |
| Ridge | 0,3037 | 0,3037 | 0,0% |

Configuração vencedora: `learning_rate=0.05`, `max_iter=500`,
`max_leaf_nodes=127`, `min_samples_leaf=20`, `l2_regularization=0.0`. Está fixada
em `train.py` para o treino ser reproduzível sem depender de rodar a busca antes.

**Dimensione o ganho antes de comemorar:** a diferença de CV é 0,0047 e o desvio
entre folds da melhor configuração é 0,0031 — cerca de 1,5 desvio. É real, mas
modesto, e bem menor que o salto de Ridge para boosting (0,08).

**O Ridge não melhorou nada.** A busca varreu `alpha` de 0,1 a 1.000 e devolveu
exatamente o default 1,0. Isso é informação, não fracasso: o gargalo do modelo
linear não é regularização, é forma funcional — ele não representa as interações
que o boosting captura.
