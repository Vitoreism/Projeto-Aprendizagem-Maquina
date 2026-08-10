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
| Observações | 15.301 anúncios com preço |
| Imóveis físicos distintos | 14.022 |
| Features | 75 na matriz → 119 depois do one-hot dentro do `Pipeline` |
| Semente | `42`, em `dataset.SEMENTE` |

O alvo vai para log porque a distribuição em reais tem **assimetria 5,92 e curtose
80,4**: a mediana é R$ 580 mil e o máximo R$ 19,8 milhões, então um punhado de
imóveis de altíssimo padrão domina o erro quadrático das 16 mil observações. Em
log a assimetria cai para −0,32. As métricas são reconvertidas para reais na
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

Resultado: **12.214 no treino, 3.087 no teste, 0 grupos dos dois lados.** O
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
custo em acurácia. Aconteceu o contrário — os números abaixo são os medidos
**naquele momento**, antes das correções que vieram depois (§8.3 e §9.7):

| Modelo | CV antes | CV logo depois |
|---|---|---|
| Gradient Boosting ajustado | 0,2232 | **0,2155** |
| Gradient Boosting (padrão) | 0,2306 | **0,2238** |
| Ridge | 0,3037 | **0,2906** |

O motivo é que os cortes globais **destruíam informação**. O encoder por fold
preserva toda categoria com suporte no treino: são **329 bairros distintos**,
contra os 38 que sobravam antes. A matriz caiu de 101 para 75 colunas, mas isso
é aparência — 26 dummies viraram 5 colunas de texto, e o que o modelo enxerga
depois da codificação é decidido no fold, não aqui.

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
| **Gradient Boosting ajustado** | **0,1998 ± 0,0032** | **R$ 171.252** | **15,8%** | **0,895** |
| Gradient Boosting (padrão) | 0,2081 ± 0,0042 | R$ 174.248 | 16,6% | 0,891 |
| Ridge | 0,2596 ± 0,0017 | R$ 252.782 | 19,8% | 0,839 |
| Baseline (mediana) | 0,6183 ± 0,0024 | R$ 443.793 | 43,1% | −0,002 |

O baseline existe como piso de sanidade: prever sempre a mediana. Seu R² de
−0,001 confirma que a montagem está correta — um baseline honesto tem que ficar
em torno de zero.

O gradient boosting ajustado erra **15,8% na mediana**. Para um imóvel de R$ 580 mil, isso
é uma faixa de cerca de R$ 100 mil — aceitável para triagem, insuficiente para
avaliação individual. O desvio da CV é pequeno (±0,004) frente à diferença entre
os modelos (0,06), então a vantagem sobre o Ridge é real, não ruído de partição.

A distância entre Ridge e boosting (0,060) indica relações não-lineares e interações
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
  10.000 amostras — temos 12.214 — e separa 10% para decidir quando parar. Essa
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

### 8.3 A segunda grade também estava mal-posta, no outro eixo

Depois que o one-hot mudou de lugar (§3.1), a busca foi refeita. Ela devolveu
`l2_regularization=1.0` e `min_samples_leaf=20` — e os dois mereciam desconfiança.

**`l2_regularization` é eixo morto.** A média entre configurações deu 0,2234 com
`l2=0` e 0,2235 com `l2=1`. A "vitória" do 1,0 aparecia só na melhor configuração
(0,2178 contra 0,2183): **0,0005**, contra um desvio entre folds de 0,0039. É
0,12 desvio — ruído de partição com cara de resultado. Fixado no default, o que
liberou metade das configurações da grade.

**`min_samples_leaf` estava na borda** — o mesmo defeito da §8.2, em outro eixo, e
que a primeira versão desta grade não pegou. Médias: 20 → 0,2204 contra
50 → 0,2265, melhorando monotonicamente até o limite inferior. Sondando abaixo:

```
2 → 0,2163    5 → 0,2154    10 → 0,2155    20 → 0,2183    50 → 0,2263
```

O mínimo real é interior, em 5–10. O comentário no código dizia que folha grande
protege contra decorar imóvel de alto padrão — verdade, mas 20 já era grande
demais e custava 0,003.

### 8.4 Resultado, e um empate resolvido fora da CV

| Modelo | CV padrão | CV ajustado | Ganho |
|---|---|---|---|
| Gradient Boosting | 0,2222 | **0,2154** | +3,1% |
| Ridge | 0,2906 | 0,2906 | 0,0% |

Com a grade corrigida, `max_leaf_nodes` ficou com ótimo interior em 127
(255 → 0,2210) e a busca está bem-posta.

A vencedora formal foi `min_samples_leaf=5` (0,2154) contra `10` (0,2155):
**0,0001 de diferença**, contra desvio de 0,0037. Pelo critério de decisão do
projeto — vantagem só é declarada acima de 0,005 — isso é **empate técnico**, e
nenhum dos critérios de desempate (explicabilidade, custo, número de
hiperparâmetros) separa os dois. O desempate foi o argumento de domínio já
documentado: cauda longa, folha maior dificulta isolar um único anúncio caro.
Ficou `min_samples_leaf=10`, e o `train.py` registra que a escolha **não** veio da
CV.

### 8.5 O ganho da busca não transferiu para o teste

Este é o resultado mais instrutivo da etapa. Comparando a configuração anterior
(`leaf=20`) com a nova (`leaf=10`), na **mesma métrica**:

| | leaf = 20 | leaf = 10 | Δ |
|---|---|---|---|
| CV MAE (log) | 0,2183 | 0,2155 | **−0,0028** |
| **Teste MAE (log)** | 0,2146 | 0,2145 | **−0,0001** |
| Teste R² (log) | 0,8680 | 0,8680 | 0,0000 |
| Teste MAE (R$) | 165.112 | 167.866 | +2.754 |
| Teste erro mediano | 16,0% | 16,2% | +0,11 p.p. |

**Apenas 4% do ganho medido na CV apareceu no teste.** A §8.1 afirmava isso em
teoria — que o score da busca é otimista porque parte da vantagem da vencedora é
sorte de partição. Aqui está a medição.

O MAE em reais até piorou. Não é contradição: em log as duas configurações são
indistinguíveis (0,2146 contra 0,2145), e a diferença em reais nasce inteira do
`exp()`, que amplifica erro na cauda cara. A árvore com folhas menores ajusta a
cauda de forma mais agressiva e generaliza um pouco pior lá.

**Por que a configuração nova ficou mesmo assim.** O protocolo é que a CV escolhe
e o teste relata. Voltar para `leaf=20` depois de ver o teste seria usar o teste
para selecionar — exatamente o que ele não pode fazer, sob pena de deixar de ser
uma estimativa honesta de generalização. O número relatado é o da configuração
escolhida antes de olhar.

**O Ridge não melhorou nada.** A busca varreu `alpha` de 0,1 a 1.000 e devolveu
exatamente o default 1,0. Isso é informação, não fracasso: o gargalo do modelo
linear não é regularização, é forma funcional — ele não representa as interações
que o boosting captura. A §9.4 mostra qual interação é essa.

### 8.6 Terceira passada, depois da canonização dos bairros

A base mudou de novo com a §9.7 — 15.583 linhas e a matriz caindo de 349 para
131 colunas depois do one-hot. A grade foi refeita.

| Modelo | CV padrão | CV ajustado | Ganho |
|---|---|---|---|
| Gradient Boosting | 0,2105 | **0,2057** | +2,3% |
| Ridge | 0,2673 | 0,2673 | 0,0% |

Configuração: `learning_rate=0.05`, `max_iter=500`, `max_leaf_nodes=127`,
`min_samples_leaf=5`, `l2_regularization=0.0`.

**`min_samples_leaf` estava na borda pela terceira vez** — no mesmo eixo que a
§8.3 já tinha corrigido uma vez. A grade ia de 5 a 50 e o ótimo caiu em 5.
Sondando abaixo:

```
1 → 0,2059    2 → 0,2050    3 → 0,2057    5 → 0,2057    10 → 0,2084
```

A curva vira em 1, então o mínimo é interior. A lição registrada: **mudou a
base, a grade precisa ser reavaliada nas bordas de novo** — o ótimo se move com
os dados, e uma grade que estava bem-posta deixa de estar. `2` e `3` entraram na
grade para que isso fique reproduzível, e não apenas num comentário.

#### A troca de `min_samples_leaf` 10 → 5, e por que ela não viola o critério

A §8.4 tinha fixado `10` por desempate de domínio. A medição na base nova
inverte, e vale explicar por quê, porque a diferença **não passa** o limiar de
0,005 do projeto.

A regra tem duas partes. A primeira diz que vence o menor MAE da CV — e é 5
(0,2057 contra 0,2084). A segunda diz quando se pode **declarar vantagem** sobre
o segundo colocado: ≥ 0,005. Os 0,0027 daqui não chegam lá. Então o que se
entrega é o vencedor da CV, sem afirmar que ele é comprovadamente melhor.

O que sustenta a escolha não é o duelo entre duas configurações, é o eixo
inteiro. A média sobre as 8 configurações de cada nível cai monotonicamente:

```
5 → 0,2089    10 → 0,2109    20 → 0,2126    50 → 0,2187
```

e os **5 folds** favorecem 5 contra 10. Isso é efeito sistemático, não sorte de
partição. O `10` tinha vindo de um argumento de domínio — cauda longa, folha
maior evita decorar imóvel caro. A medição contradiz o *prior* nesta faixa, e o
*prior* cede.

Por que `5` e não `2`, que é o mínimo da sonda (0,2050): 2, 3 e 5 estão dentro de
0,0007 uns dos outros. Empate de verdade, desempatado pela folha maior — o ponto
mais conservador da faixa, ao mesmo custo em CV.

#### Desta vez o ganho transferiu

A §8.5 mostrou um ajuste em que só 4% do ganho da CV apareceu no teste. Aqui:

| | leaf = 10 | leaf = 5 | Δ |
|---|---|---|---|
| CV MAE (log) | 0,2084 | 0,2057 | −0,0027 |
| Teste MAE (log) | 0,2089 | 0,2062 | **−0,0027** |
| Teste R² (log) | 0,884 | 0,887 | +0,003 |
| Teste MAE (R$) | 165.714 | 164.529 | −1.185 |

Transferência de **100%**, contra 4% da vez anterior. Não há o que comemorar
numa amostra de dois casos — mas registra que "o ganho da busca é sempre
otimista" é tendência, não lei, e que o teste continua sendo a única forma de
saber qual dos dois aconteceu.

### 8.7 Quarta passada: a primeira vez que a grade estava bem-posta

O descarte dos repasses (§9.10) mudou a base de novo, então o eixo que já caiu na
borda três vezes foi remedido antes de qualquer conclusão:

```
1 → 0,1998   2 → 0,1997   3 → 0,1989   5 → 0,1998
8 → 0,2006  10 → 0,2022  15 → 0,2024  20 → 0,2033
```

Pela primeira vez a curva sobe dos **dois** lados: o mínimo está em 3, interior, e
não encostado numa ponta. Não há grade para estender.

**E não há motivo para mudar nada.** O valor em produção é 5, a 0,0009 do mínimo
— e 1, 2, 3 e 5 cabem todos dentro desses mesmos 0,0009. É região plana, não
diferença; o limiar de 0,005 do projeto está cinco vezes acima disso. Trocar 5
por 3 seria perseguir a quarta casa decimal de uma partição específica.

Fica a regra, que já valia três passadas atrás e agora tem contraexemplo: **quando
a base muda, as bordas da grade precisam ser reavaliadas** — o ótimo se move com
os dados. A novidade é que remedir também pode concluir *nada a fazer*, e essa é
uma resposta tão boa quanto as outras.

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
média = −0,0036   mediana = −0,0007   desvio = 0,2854   assimetria = +0,03
```

Em log, a mediana do resíduo é praticamente zero: o modelo não tem viés global.
A assimetria residual de +0,03 é o que sobrou da assimetria 5,92 do alvo em
reais — a transformação log fez praticamente todo o trabalho.

### 9.2 Onde o modelo erra: nas duas pontas, não só no topo

| Faixa de preço | n | Preço mediano | Viés | Erro mediano | MAE |
|---|---|---|---|---|---|
| Q1 (mais barato) | 618 | R$ 180 mil | +7,8% | 16,8% | R$ 50 mil |
| Q2 | 627 | R$ 406 mil | +8,5% | **13,6%** | R$ 79 mil |
| Q3 | 612 | R$ 589 mil | +1,9% | 14,2% | R$ 103 mil |
| Q4 | 620 | R$ 804 mil | −3,9% | 14,0% | R$ 145 mil |
| Q5 (mais caro) | 609 | R$ 1,4 mi | **−15,2%** | **20,1%** | R$ 450 mil |

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
conservadora, em cerca de 15%.

Eu previa que o erro se concentraria no alto padrão, onde os dados são esparsos.
Certo em parte. O Q5 é de longe o pior (20,1% contra 13,6% no melhor quintil). O Q1 era
quase tão ruim quanto — mas deixou de ser depois que os bairros foram
canonizados (§9.7): caiu de 20,0% para 16,8%, porque boa parte do erro ali não
era escassez de dado, era bairro errado.

O erro em reais, esse sim, cresce monotonicamente: R$ 50 mil no Q1 contra R$ 450
mil no Q5. Para triagem em massa a leitura relevante é a percentual; para decidir
sobre um imóvel específico, a em reais.

### 9.3 Importância por permutação: o modelo depende de duas coisas

Atributo inteiro permutado (não dummy a dummy — permutar uma dummy de bairro por
vez deixaria as outras 328 entregando a resposta e a importância sairia zero por
construção). Unidade: quanto o MAE em log piora ao embaralhar a coluna.

| Atributo | Importância | Desvio |
|---|---|---|
| `bairro` | **+0,2580** | 0,0043 |
| `area_util` | **+0,1797** | 0,0054 |
| `garagens` | +0,0655 | 0,0030 |
| `suites` | +0,0271 | 0,0025 |
| `origem_anuncio` | +0,0222 | 0,0014 |
| `quartos` | +0,0158 | 0,0008 |
| `condominio` | +0,0152 | 0,0011 |
| `banheiros` | +0,0098 | 0,0008 |
| `area_total` | +0,0081 | 0,0008 |
| `com_varanda_gourmet` | +0,0054 | 0,0006 |

Localização e tamanho respondem por **0,44 dos 0,65** de importância total.
Embaralhar `bairro` sozinho piora o MAE em 0,2580, mais do que o MAE final
inteiro (0,2062) — sem bairro o modelo é pior que inútil.

A distância entre os dois cresceu depois da canonização (§9.7): `bairro` subiu
de +0,2226 para +0,2580 e `area_util` caiu de +0,2001 para +0,1797. Faz sentido
— parte do sinal de localização estava perdida em rótulo errado, e o modelo a
compensava pela área.

**29 dos 75 atributos têm importância indistinguível de zero.** Quase todos são
comodidades. Isso não significa que comodidade não importe: significa que, dado
o bairro e a área, ela não acrescenta.

Nota de honestidade sobre `origem_anuncio`: está na matriz como controle do
artefato de portal, e a permutação confirma que o modelo a usa (+0,0222). Isso é
esperado e é justamente por isso que ela fica — sem a coluna, a diferença entre
portais seria absorvida como se fosse diferença entre imóveis.

### 9.4 Correlação × importância: onde os dois discordam

A correlação é bivariada, o modelo é multivariado. Onde as duas listas divergem
está o que só um dos métodos enxerga.

**A correlação prometia mais do que o modelo usa:**

| Feature | \|Spearman\| | Importância | Salto de posto |
|---|---|---|---|
| `com_closet` | 0,178 | −0,0000 | −94 |
| `com_quadra_esportiva` | 0,151 | +0,0001 | −80 |
| `com_copa` | 0,128 | −0,0000 | −72 |

`com_closet` está entre as 10 maiores correlações com o preço e vale **zero**
para o modelo. Não é contradição: closet aparece em apartamento grande de bairro
caro. Dado `area_util` e `bairro`, ele não acrescenta nada — a correlação estava
medindo o efeito de outra variável através dele.

**O modelo usa mais do que a correlação sugeria:**

| Feature | \|Spearman\| | Importância | Salto de posto |
|---|---|---|---|
| `bairro_bessa` | 0,010 | +0,0112 | +98 |
| `bairro_aeroclube` | 0,038 | +0,0030 | +65 |

`bairro_bessa` tem correlação **0,010** com o preço — praticamente nada — e ainda
assim é uma das dummies mais úteis do modelo. O Bessa tem preço mediano de
R$ 570 mil, colado na mediana geral de R$ 575 mil, então a correlação linear com
o preço é nula por construção. O que o bairro informa é a relação **preço por
m²** dentro dele, que só existe em interação com `area_util`. Nenhuma correlação
bivariada consegue ver isso.

**Esta é a resposta prática de por que o boosting ganha do Ridge por 0,07.**

### 9.5 O que o resíduo revelou sobre os dados

Os 19 anúncios do teste abaixo de R$ 50 mil têm erro mediano de **+75%** — o
modelo prevê consistentemente muito acima. Não é falha do modelo: R$ 35 mil por
58 m² dá R$ 603/m², contra uma mediana de **R$ 9.045/m²** na base. São entradas
de financiamento, permutas ou erro de digitação anunciados como preço de venda.

Dos 39 anúncios com erro acima de 100%, **33% têm preço/m² abaixo de R$ 1.500**.

O piso de plausibilidade em `build_features` está em R$ 20.000, permissivo
demais. Um piso por **preço/m²** — e não por preço absoluto — pegaria esses casos
sem descartar imóvel pequeno legitimamente barato. Fica registrado como próximo
passo; não foi aplicado agora porque mudaria a base sob os números já relatados
nesta etapa.

> Aplicado depois, na seção **9.10** — onde a investigação mostrou que a hipótese
> acima estava certa na conclusão e errada na causa: não era erro de digitação,
> era outro produto vendido com a mesma etiqueta.

Dois outros segmentos, para fechar:

- **Portal.** `zapimoveis` 15,7%, `chaves_na_mao` 15,9%, anúncios presentes nos
  dois 14,6%. O imóvel que aparece nos dois portais é mais fácil de prever, o que
  faz sentido: são os anúncios com ficha mais completa.
- **Campos ausentes.** A correlação de Spearman entre número de campos numéricos
  em branco e erro absoluto é **0,038**. Praticamente nula — a imputação com
  indicadora está segurando bem a ausência, e o erro não vem de lá.

### 9.7 A canonização dos bairros

A análise por segmento levantou uma pergunta: os bairros estão certos? Não
estavam. **14,0% dos anúncios tinham o bairro errado**, e `bairro` é o atributo
mais importante do modelo.

#### Os três defeitos

A função `extrair_bairro` jogava o endereço inteiro em `normalizar_texto`, que
troca vírgula e hífen por espaço. Isso **achatava a estrutura** do endereço num
fluxo único de palavras e destruía a única pista confiável de onde o campo do
bairro começa e termina. Sem essa pista, o que sobrou foram dois paliativos e um
erro de ordenação:

**1. Casamento por substring, na ordem em que a lista foi escrita.** A lista
tinha `"cabo branco"` antes de `"altiplano"`, então **"Altiplano Cabo Branco"
casava com Cabo Branco**:

| Rotulados `cabo_branco` | n | Preço/m² mediano |
|---|---|---|
| de fato Cabo Branco | 1.573 | R$ 15.000 |
| de fato Altiplano Cabo Branco | **511** | R$ 11.156 |

26% de diferença fundida numa categoria só. O mesmo com Tambaú/Tambauzinho: 119
anúncios, 31% de diferença. É o pior tipo de erro possível aqui — não embaralha,
**junta duas faixas de preço sob o mesmo rótulo**.

**2. O fallback inventava bairros.** Endereço que não batia com nenhum dos 30
nomes da lista virava *a primeira palavra com mais de 3 letras*. Daí saíram
`avenida`, `doutor`, `professor`, `comerciante`, `aposentada`, `telegrafista` —
1.589 anúncios (9,8%) em categorias que não existem. O balde `avenida` tinha IQR
de preço/m² de **R$ 6.898**, contra R$ 1.316 de um bairro real: agrupava imóveis
de toda a cidade.

**3. Nomes não-canônicos.** `geisel` em vez de Ernesto Geisel, `planalto` em vez
de Planalto Boa Esperança, `valentina` em vez de Valentina de Figueiredo.

#### A correção

O endereço tem estrutura, e os dois portais escrevem diferente:

```
chaves na mao:  'Rua X, 155, Jardim Oceania,João Pessoa/PB'
zapimoveis:     'Rua X, 38 - Bessa, João Pessoa - PB'
```

A nova `extrair_bairro` quebra por vírgula **e** hífen, percorre os campos do fim
para o começo (o bairro fica antes da cidade) e casa contra os **64 bairros
oficiais** de João Pessoa, por conjunto de tokens — ignorando artigos, para que
"Valentina Figueiredo" e "Valentina de Figueiredo" sejam o mesmo lugar.

**A lista de referência não é o `neighborhoods.csv`.** Aquele arquivo mistura
bairro oficial com loteamento, conjunto e até praia de outro município
(Camboinha é Cabedelo, Carapibus é Conde), e usa nomes que não são os oficiais:
traz "Altiplano Cabo Branco" onde o bairro se chama **Altiplano**, e "José
Américo de Almeida" onde se chama **José Américo**. A lista fechada dos 64 está
no código.

Duas decisões que valem justificativa:

- **Vence o nome mais específico**, não o primeiro de uma lista. É literalmente a
  correção do defeito 1. Há uma sutileza: o bairro oficial se chama *Altiplano*,
  de um token só, e sozinho ele **perderia** de `{cabo, branco}`, de dois — o bug
  voltaria. Por isso `"altiplano cabo branco"` está no mapa de apelidos e entra
  na ordenação por especificidade com três tokens, vencendo os dois.
- **Nunca inventa.** O que não casa vira `nao_informado`. Sobraram **13 anúncios**
  (0,08%): 8 sem endereço utilizável e 5 fora do município (praias de Camboinha e
  Carapibus, que são Cabedelo e Conde).

#### Sete localidades que não estão entre as 64

Jardim Luna (43 anúncios), Novo Milênio (7), Colinas do Sul (6), Conjunto
Esplanada, Cidade Verde, Jardim Planalto e Jardim das Acácias aparecem nos
anúncios e não constam da lista oficial. A primeira versão desta correção
mandou as sete para `nao_informado`, por não ter como verificá-las.

Confirmação de quem conhece a cidade: funcionam como bairro. E os dados
sustentam — a mesma medida de dispersão que expôs os baldes falsos:

| | n | Mediana R$/m² | IQR | CV |
|---|---|---|---|---|
| **Jardim Luna** | 41 | **10.132** | 2.649 | **0,23** |
| Novo Milênio | 5 | 3.100 | 2.629 | 0,72 |
| Colinas do Sul | 6 | 2.564 | 1.320 | 0,49 |
| *bairro oficial típico* | — | 6.082 | 1.964 | 0,35 |
| *`avenida`, o balde antigo* | 29 | 4.091 | **6.898** | — |

Jardim Luna é **mais homogêneo que o bairro oficial mediano** (CV 0,23 contra
0,35) e nitidamente mais caro que a cidade. É uma área coerente, nada parecida
com o balde `avenida`, que misturava imóveis de toda a cidade. Mandá-la para
`nao_informado` descartava sinal bom, e ainda a misturava com anúncios que
simplesmente não têm endereço.

As sete entraram numa lista à parte, `LOCALIDADES_RECONHECIDAS`. A garantia de
"nunca inventa" continua de pé: as duas listas são curadas e fechadas — o que o
fallback antigo fazia era criar categoria a partir de qualquer palavra solta.

**Efeito no modelo: nenhum.** CV 0,2085 → 0,2084, dentro do ruído (medido antes
do reajuste de hiperparâmetros da §8.6). São 60
anúncios em 15.583, e o `min_frequency=30` do `Pipeline` só deixa Jardim Luna
virar coluna própria; as outras seis viram categoria rara dentro do fold. A
justificativa aqui é de correção, não de métrica — igual ao resto desta seção.

Resultado: **329 valores distintos → 66** (65 lugares com anúncio, mais
`nao_informado`). Nenhum valor fora da lista, e um teste varre a base inteira a
cada execução para garantir isso.

#### Efeito colateral na deduplicação

`bairro_norm` faz parte da chave de deduplicação `(bairro, preço, área,
quartos)`. Com bairros errados, o mesmo imóvel anunciado nos dois portais podia
receber rótulos diferentes de cada lado e **escapar da fusão**. Rerodando:

```
22.599 anúncios brutos − 7.016 duplicatas fundidas = 15.583 únicos
```

São **579 duplicatas a mais** encontradas, e a base cai de 16.162 para 15.583
linhas. Nada foi perdido: são anúncios do mesmo imóvel finalmente unificados.

#### Quanto isso valeu — medido de forma isolada

A base mudou junto com o bairro, então comparar as duas execuções não isola nada.
O A/B roda sobre as **mesmas linhas e as mesmas folds**, trocando só a coluna:

| | CV MAE (log) | folds |
|---|---|---|
| bairro canônico | **0,2097** ± 0,0029 | 0,2139 · 0,2047 · 0,2093 · 0,2101 · 0,2105 |
| rótulo antigo | 0,2144 ± 0,0029 | 0,2145 · 0,2107 · 0,2119 · 0,2183 · 0,2169 |

**Os cinco folds favorecem o canônico**, com diferença média de **0,0047**. Isso
decompõe a melhora total: 0,2155 → 0,2144 é a mudança de base (0,0011), e
0,2144 → 0,2097 é o bairro (0,0047). Fechar a lista nos 64 oficiais e reajustar
os hiperparâmetros sobre a base nova levaram a CV a **0,2057** (§8.6).

A hipótese registrada antes de rodar era "melhora pequena, entre 0,002 e 0,008".
Confirmou.

**Nota sobre o critério de decisão.** O limiar declarado do projeto é 0,005 para
declarar vantagem, e 0,0047 fica logo abaixo. Vale ser explícito: aquele limiar
existe para proteger contra viés de seleção ao escolher o máximo de N
configurações, e aqui não há N — é uma comparação única, planejada, com 5 de 5
folds concordando. Mais importante: **a justificativa da correção não é o MAE.**
14% dos anúncios tinham bairro errado; consertar isso estaria certo mesmo que o
MAE não mexesse.

#### Efeitos secundários

| | antes | depois |
|---|---|---|
| Colunas após o one-hot | 349 | **131** |
| Importância de `bairro` | +0,2226 | **+0,2580** |
| Assimetria do resíduo | +0,42 | **+0,03** |
| Erro mediano no Q1 | 20,0% | **16,8%** |
| Erro mediano no teste | 16,2% | **15,4%** |
| R² (log) no teste | 0,868 | **0,887** |

A queda de 349 para 131 colunas com o modelo *melhorando* é o argumento da §10 do
material da disciplina — menos atributos, mesmo resultado ou melhor, com menor
custo — só que aqui o corte não veio de seleção de features, e sim de parar de
gerar categorias falsas.

O Q1 melhorou bastante: boa parte do erro nos imóveis baratos não era escassez de
dado, era bairro errado. Gramame, o bairro com maior correlação negativa com
preço, tinha 60 anúncios espalhados nos baldes `inacio` e `josinaldo`.

#### Uma previsão que não se confirmou

Eu previa que `cabo_branco` melhoraria, "porque hoje são dois bairros somados".
Não melhorou — foi de 16,4% para **17,7%** de erro mediano.

O motivo é instrutivo. Depois de puro, Cabo Branco tem IQR de preço/m² de **R$
7.882**, o maior de todos os bairros (o típico é R$ 1.446). É a orla de alto
padrão: mesmo bairro, imóveis de valores radicalmente diferentes. Misturar o
Altiplano, mais homogêneo, *diluía* a dificuldade e fazia o número parecer melhor.

Purificar uma categoria não a torna mais fácil — torna o número honesto. O ganho
real apareceu em outro lugar: no Q1, onde o erro caiu 2,7 pontos.

### 9.8 Figuras

| Arquivo | Conteúdo |
|---|---|
| `docs/figuras/residuos_diagnostico.png` | 4 painéis: resíduo × previsto, distribuição, erro por faixa, previsto × real |
| `docs/figuras/importancia_permutacao.png` | top 20 por permutação, com barra de erro |

### 9.9 Execução

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.models.analysis
```

Saídas em `data/processed/`: `residuos_teste.csv`, `residuos_por_segmento.csv`,
`importancia_permutacao.csv`, `importancia_permutacao_codificada.csv` (esta com o
`|Spearman|` ao lado, para o confronto de §9.4).

### 9.10 O preço que não era preço

A §9.5 registrou 19 anúncios com erro de +75% e propôs um piso por preço/m².
Ao abrir os anúncios para escolher o valor do piso, apareceu a causa — e ela não
era a suposta:

> *"Repasse no Valentina: Chaves R$ 21.500 mil e Parcela Menor que Aluguel (R$ 719)"*

Não é erro de digitação. É **repasse (ágio)**: o anunciante vende a posição dele
num financiamento, e o valor da etiqueta é o que se paga pelas chaves — o
comprador ainda assume as parcelas restantes. São dois produtos diferentes com o
mesmo rótulo `preco_venda`. Mantidos na base, ensinam que um apartamento de dois
quartos no Gramame vale R$ 22.000.

**Por que o corte não é onde a intuição colocaria.** A distribuição global de
preço/m² não tem vale: é contínua de R$ 250 a R$ 4.000. Então o piso é escolha,
não descoberta — e foi calibrado contra um sinal *independente*, a palavra
"repasse" ou "ágio" no texto do anúncio, que 177 anúncios declaram:

| piso | descartados | declaram repasse | precisão |
|---|---|---|---|
| R$ 500 | 31 | 25 | 81% |
| **R$ 1.000** | **111** | **84** | **76%** |
| R$ 1.500 | 210 | 104 | 50% |
| R$ 2.000 | 440 | 111 | 25% |

A precisão desaba entre R$ 1.000 e R$ 1.500. A queda tem nome: ali entra uma
segunda população, **legítima**. São 311 anúncios de venda direta/leilão da
Caixa, com preço/m² concentrado entre R$ 1.164 (p05) e R$ 1.899 (p75) — vendas
reais num regime de preço deprimido, não erro. O piso em R$ 2.000, que a leitura
ingênua da §9.5 sugeria, teria apagado todas elas.

O piso fica **abaixo** do p05 desse grupo, de propósito. Custo assumido e
declarado: 21 anúncios de venda direta caem junto, 19% dos descartes. É o preço
de não fazer uma regra de validade de dado depender de texto de marketing.

O texto **calibrou** o piso; não filtra nada. Usar "repasse" como filtro
descartaria anúncio comum que só cita a palavra no rodapé da imobiliária — 78
dos 177 têm preço/m² acima de R$ 1.250 e são anúncios normais.

#### A área é outro defeito, e pede outro remédio

Nove anúncios têm preço/m² baixo por um motivo diferente: a **área** está errada,
o preço está certo. É o separador decimal perdido no scrape — 988 m² num anúncio
cujo próprio título diz *"Vista para o Mar em Manaíra | 98m²"*, e 1.372 m² em
outro que se anuncia como *"137,20 m² e 3 quartos"*.

Aqui o vale que falta na distribuição global **existe**: entre os 85 anúncios com
mais de 400 m², o preço/m² pula de R$ 1.666 para R$ 3.288 sem nada no meio.
Qualquer limiar entre 1.700 e 3.200 separa os mesmos nove anúncios.

Os dois lados do vale são reconhecíveis. Abaixo: 1.280 m² de 3 quartos em Tambaú
por R$ 550.000. Acima: coberturas de 5 quartos no Cabo Branco e no Miramar, de
664 e 1.260 m², por R$ 11 e R$ 19,8 milhões — que são **reais**. Um teto absoluto
de área mataria justamente os imóveis mais caros da base. Por isso a regra olha o
par (área, preço/m²), nunca a área sozinha.

O remédio também difere: aqui **anula-se a área**, não a linha. O preço de
R$ 1.019.334 no Jardim Oceania é dado bom; só a área de 984 m² é que é 98,4.

E a ordem importa. A regra da área roda **antes** do piso: sem área, esses
anúncios deixam de ter preço/m² e saem da mira do descarte seguinte — que é o
que se quer, porque o problema deles nunca foi o preço. Rodando na ordem
inversa, os R$ 550.000 de Tambaú sairiam da base junto com a área errada.

Ambas as regras rodam **depois** do enriquecimento pela descrição: se o pipeline
recupera "98 m²" do texto, a regra nem precisa disparar. Julgar antes puniria o
anúncio que o próprio pipeline ia consertar.

#### O ganho é quase todo ilusório, e mesmo assim vale

A base foi de 15.583 para **15.476** linhas (107 descartes, 9 áreas anuladas) e o
número relatado melhorou bastante:

| | antes | depois |
|---|---|---|
| CV MAE (log) | 0,2057 | **0,1998** |
| R² (log) | 0,887 | **0,895** |

Mas **esses dois números não são comparáveis**: mudaram a base *e* o conjunto de
teste. A pergunta que importa — a presença do ágio no treino piora a previsão dos
anúncios legítimos? — exige avaliar os dois modelos nas mesmas linhas.

Split calculado uma vez sobre as 15.292 linhas idênticas nas duas bases, mesmo
teste de 3.087 anúncios, única diferença sendo os 107 de ágio no treino de A:

| | CV | teste MAE(log) | erro mediano | R² |
|---|---|---|---|---|
| A: com ágio no treino | 0,2091 | 0,2035 | 15,37% | 0,8925 |
| B: sem ágio no treino | 0,2002 | **0,2005** | 15,47% | **0,8971** |

O efeito real é **+0,0031** de MAE(log) e **+0,0046** de R². O erro mediano fica
0,10 p.p. *pior*, o que é ruído. Ou seja: dos 0,0059 de melhora relatada, cerca
de metade é o modelo prevendo melhor e o resto é o conjunto de avaliação ter
perdido linhas que eram impossíveis por construção.

E 0,0031 **não passa** o limiar de 0,005 do projeto. Então a afirmação correta
não é "o modelo melhorou": é que **a base ficou certa**. Aqueles 107 anúncios não
tinham preço de venda para prever, e um modelo julgado por acertá-los estava
sendo julgado pela pergunta errada.

É a mesma lição da §9.7 pelo avesso. Lá, purificar uma categoria piorou o número
e o tornou honesto. Aqui, remover registros inválidos melhorou o número — e a
parte do ganho que é honesta é a menor.

Sobra registrado: a venda direta/leilão é uma população de 311 anúncios com
regime de preço próprio, e hoje o modelo não sabe distinguí-la. Uma binária
`venda_direta` é candidata natural a atributo — não foi criada agora para não
misturar mudança de dado com mudança de modelo na mesma medição.
