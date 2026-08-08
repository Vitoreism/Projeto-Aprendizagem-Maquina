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
| Features | 75 na matriz → 349 depois do one-hot dentro do `Pipeline` |
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
├── numéricas (8):   SimpleImputer(mediana, add_indicator) → StandardScaler
├── binárias (62):   passthrough
└── nominais (5):    OneHotEncoder(min_frequency=30,
                                   handle_unknown='infrequent_if_exist')
```

Imputação, escala **e o one-hot** não acontecem antes do split. Os três moram
dentro do `Pipeline`, então o `fit` é refeito em cada fold da validação cruzada e
nenhuma estatística do fold de validação entra no de treino.

Três decisões que valem justificativa:

- **`add_indicator=True`.** Com `iptu` ausente em 80% dos anúncios, `area_total`
  em 61% e `suites`/`condominio` perto de 47% e 54%, o próprio silêncio do anunciante é
  informação. A indicadora preserva isso; imputar sem ela apagaria o sinal.
- **Mediana, não média.** Pelo mesmo motivo do log: as caudas são pesadas.
- **`OneHotEncoder` no `Pipeline`, não `get_dummies` na matriz.** Ver §3.1.

As binárias passam direto — já estão em 0/1, e padronizá-las só destruiria a
interpretação sem ganho.

### 3.1 Por que o one-hot saiu do `build_features`

Enquanto as nominais viravam dummies com `pd.get_dummies` sobre a base inteira,
duas decisões eram tomadas contando linhas que virariam teste:

1. bairro com menos de 30 imóveis colapsava em `outros`;
2. dummy com frequência abaixo de 1% era descartada.

Nenhuma das duas olha o alvo, então não é vazamento de alvo — é **vazamento
estrutural**: o conjunto de colunas é definido usando o teste. A correção previa
custo em acurácia. Aconteceu o contrário:

| Modelo | CV antes | CV depois |
|---|---|---|
| Gradient Boosting ajustado | 0,2232 | **0,2183** |
| Gradient Boosting (padrão) | 0,2306 | **0,2238** |
| Ridge | 0,3037 | **0,2906** |

O motivo é que os cortes globais **destruíam informação**. O encoder por fold
preserva toda categoria com suporte no treino: são **329 bairros distintos**,
contra os 38 que sobravam antes. A matriz caiu de 101 para 75 colunas, mas isso
é aparência — 26 dummies viraram 5 colunas de texto e o modelo passou a enxergar
349 colunas depois da codificação, não menos.

`handle_unknown='infrequent_if_exist'` manda bairro que só aparece no teste para
o mesmo balde das categorias raras, em vez de quebrar ou de virar coluna que o
modelo nunca viu.

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
| **Gradient Boosting ajustado** | **0,2183 ± 0,0041** | **R$ 165.112** | **16,0%** | **0,868** |
| Gradient Boosting (padrão) | 0,2238 ± 0,0032 | R$ 171.392 | 17,0% | 0,861 |
| Ridge | 0,2906 ± 0,0045 | R$ 290.742 | 22,3% | 0,766 |
| Baseline (mediana) | 0,6531 ± 0,0047 | R$ 417.354 | 42,5% | −0,001 |

O baseline existe como piso de sanidade: prever sempre a mediana. Seu R² de
−0,001 confirma que a montagem está correta — um baseline honesto tem que ficar
em torno de zero.

O gradient boosting ajustado erra **16,0% na mediana**. Para um imóvel de R$ 575 mil, isso
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

---

## 9. Análise de resíduos e importância por permutação

`src/imoveis_jp/models/analysis.py`. Tudo medido no **conjunto de teste**, sobre o
`gradient_boosting_ajustado`.

A escolha do teste é deliberada: importância medida no treino responde "do que o
modelo se lembrou", e a pergunta do projeto é "do que ele precisa para acertar
num imóvel que nunca viu". O preço é que o teste já foi usado para relatar a
métrica final — por isso **nada daqui volta para o modelo como seleção de
atributo**. É leitura, não decisão.

### 9.1 Os resíduos estão centrados

```
média = +0,0022   mediana = +0,0001   desvio = 0,2988   assimetria = +0,33
```

Em log, a mediana do resíduo é praticamente zero: o modelo não tem viés global.
A assimetria residual de +0,33 é o que sobrou da assimetria 5,92 do alvo em
reais — a transformação log fez quase todo o trabalho.

### 9.2 Onde o modelo erra: nas duas pontas, não só no topo

| Faixa de preço | n | Preço mediano | Viés | Erro mediano | MAE |
|---|---|---|---|---|---|
| Q1 (mais barato) | 638 | R$ 190 mil | **+6,3%** | **19,0%** | R$ 55 mil |
| Q2 | 632 | R$ 410 mil | +9,1% | 14,8% | R$ 88 mil |
| Q3 | 639 | R$ 590 mil | +0,8% | 13,2% | R$ 103 mil |
| Q4 | 633 | R$ 830 mil | −3,8% | 15,1% | R$ 158 mil |
| Q5 (mais caro) | 625 | R$ 1,4 mi | **−13,5%** | **19,2%** | R$ 426 mil |

O viés troca de sinal monotonicamente do Q1 ao Q5: **o modelo puxa tudo para o
meio.** É a regressão à média clássica de um estimador que minimiza erro — ele
prefere errar pouco em muitos imóveis a acertar os extremos.

Vale reparar numa aparente contradição com a figura: no painel "resíduo × valor
previsto" a mediana por vintil é uma linha reta em zero, sem viés nenhum. Os dois
resultados estão certos e medem coisas diferentes — o painel condiciona no
**previsto**, a tabela condiciona no **real**. Um modelo não-enviesado dado o que
ele mesmo previu ainda é enviesado dado a verdade, e essa diferença *é* a
regressão à média. Diagnosticar só pelo gráfico clássico esconderia o efeito.

Consequência prática: a previsão do modelo não deve ser lida como estimativa
pontual de um imóvel de alto padrão. Nesse segmento ela é sistematicamente
conservadora, em cerca de 13%.

Eu previa que o erro se concentraria no alto padrão, onde os dados são esparsos.
Metade certo: o Q5 é de fato o pior, mas o Q1 empata com ele (19,0% contra
19,2%) por um motivo diferente — não é escassez, é **qualidade de dado** (§9.5).

O erro em reais, esse sim, cresce monotonicamente: R$ 55 mil no Q1 contra R$ 426
mil no Q5. Para triagem em massa a leitura relevante é a percentual; para decidir
sobre um imóvel específico, a em reais.

### 9.3 Importância por permutação: o modelo depende de duas coisas

Atributo inteiro permutado (não dummy a dummy — permutar uma dummy de bairro por
vez deixaria as outras 328 entregando a resposta e a importância sairia zero por
construção). Unidade: quanto o MAE em log piora ao embaralhar a coluna.

| Atributo | Importância | Desvio |
|---|---|---|
| `bairro` | **+0,2166** | 0,0054 |
| `area_util` | **+0,1989** | 0,0028 |
| `garagens` | +0,0602 | 0,0020 |
| `suites` | +0,0254 | 0,0018 |
| `condominio` | +0,0175 | 0,0009 |
| `quartos` | +0,0140 | 0,0011 |
| `area_total` | +0,0100 | 0,0009 |
| `origem_anuncio` | +0,0096 | 0,0008 |
| `banheiros` | +0,0077 | 0,0007 |
| `com_varanda_gourmet` | +0,0047 | 0,0007 |

Localização e tamanho respondem por **0,42 dos 0,60** de importância total. Cada
um deles vale, sozinho, o dobro do MAE final inteiro (0,2146) — embaralhar
qualquer um dos dois destrói o modelo.

**28 dos 75 atributos têm importância indistinguível de zero.** Quase todos são
comodidades. Isso não significa que comodidade não importe: significa que, dado
o bairro e a área, ela não acrescenta.

Nota de honestidade sobre `origem_anuncio`: ele está na matriz como controle do
artefato de portal, e a permutação confirma que o modelo o usa (+0,0096). Isso é
esperado e é justamente por isso que ele fica — sem a coluna, a diferença entre
portais seria absorvida como se fosse diferença entre imóveis.

### 9.4 Correlação × importância: onde os dois discordam

A correlação é bivariada, o modelo é multivariado. Onde as duas listas divergem
está o que só um dos métodos enxerga.

**A correlação prometia mais do que o modelo usa:**

| Feature | \|Spearman\| | Importância | Salto de posto |
|---|---|---|---|
| `com_lavabo` | 0,228 | +0,0000 | −92 |
| `com_closet` | 0,177 | −0,0001 | −90 |
| `com_playground` | 0,174 | +0,0001 | −59 |
| `com_gerador_de_energia` | 0,131 | −0,0000 | −71 |

`com_lavabo` é a 11ª maior correlação com o preço e vale **zero** para o modelo.
Não é contradição: lavabo correlaciona com preço porque aparece em apartamento
grande de bairro caro. Dado `area_util` e `bairro`, ele não acrescenta nada — a
correlação estava medindo o efeito de outra variável através dele.

**O modelo usa mais do que a correlação sugeria:**

| Feature | \|Spearman\| | Importância | Salto de posto |
|---|---|---|---|
| `bairro_bessa` | 0,012 | +0,0064 | +98 |
| `bairro_aeroclube` | 0,039 | +0,0020 | +65 |

`bairro_bessa` tem correlação **0,012** com o preço — praticamente nada — e ainda
assim é uma das dummies mais úteis do modelo. O Bessa tem preço mediano de
R$ 570 mil, colado na mediana geral de R$ 575 mil, então a correlação linear com
o preço é nula por construção. O que o bairro informa é a relação **preço por
m²** dentro dele, que só existe em interação com `area_util`. Nenhuma correlação
bivariada consegue ver isso.

**Esta é a resposta prática de por que o boosting ganha do Ridge por 0,07.**

### 9.5 O que o resíduo revelou sobre os dados

Os 16 anúncios do teste abaixo de R$ 50 mil têm erro mediano de **+136%** — o
modelo prevê consistentemente muito acima. Não é falha do modelo: R$ 35 mil por
58 m² dá R$ 603/m², contra uma mediana de **R$ 9.019/m²** na base. São entradas
de financiamento, permutas ou erro de digitação anunciados como preço de venda.

Dos 53 anúncios com erro acima de 100%, **42% têm preço/m² abaixo de R$ 1.500**.

O piso de plausibilidade em `build_features` está em R$ 20.000, permissivo
demais. Um piso por **preço/m²** — e não por preço absoluto — pegaria esses casos
sem descartar imóvel pequeno legitimamente barato. Fica registrado como próximo
passo; não foi aplicado agora porque mudaria a base sob os números já relatados
nesta etapa.

Dois outros segmentos, para fechar:

- **Portal.** `zapimoveis` 15,9%, `chaves_na_mao` 17,3%, anúncios presentes nos
  dois 14,5%. O imóvel que aparece nos dois portais é mais fácil de prever, o que
  faz sentido: são os anúncios com ficha mais completa.
- **Campos ausentes.** A correlação de Spearman entre número de campos numéricos
  em branco e erro absoluto é **0,051**. Praticamente nula — a imputação com
  indicadora está segurando bem a ausência, e o erro não vem de lá.

### 9.6 Figuras

| Arquivo | Conteúdo |
|---|---|
| `docs/figuras/residuos_diagnostico.png` | 4 painéis: resíduo × previsto, distribuição, erro por faixa, previsto × real |
| `docs/figuras/importancia_permutacao.png` | top 20 por permutação, com barra de erro |

### 9.7 Execução

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.models.analysis
```

Saídas em `data/processed/`: `residuos_teste.csv`, `residuos_por_segmento.csv`,
`importancia_permutacao.csv`, `importancia_permutacao_codificada.csv` (esta com o
`|Spearman|` ao lado, para o confronto de §9.4).
