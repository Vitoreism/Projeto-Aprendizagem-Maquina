# Roteiro de apresentação — metodologia do projeto

**Projeto:** Previsão e análise de preços de apartamentos em João Pessoa (PB)
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB
**Estado do repo:** `main` em `d092478` (issue #25 fechada — comparação final, PCA e t-SNE)

> **Como usar.** As seções 1, 5 e 7 são para **todo mundo decorar**. A seção 4 é o
> roteiro de fala dividido por pessoa. A seção 6 é o treino para a situação temida:
> *a professora pergunta da tarefa X para quem fez Y* — cada resposta ali está em
> 2–3 frases, no nível que qualquer membro sustenta sem ter escrito o código.
>
> **Regra de ouro:** ninguém precisa saber o código dos outros. Todo mundo precisa
> saber **a decisão, o número e o porquê** de cada etapa.

---

## 1. O pitch de 60 segundos (todos decoram)

> Coletamos ~22,6 mil anúncios de apartamentos à venda em João Pessoa de dois
> portais (chavesnamao e ZapImóveis). Depois de deduplicar, sobraram **15.476
> imóveis** com **76 atributos**. O alvo é **log do preço** — em reais a
> distribuição tem assimetria 5,92 e um imóvel de R$ 19,8 milhões dominaria o erro.
> O split é **80/20 agrupado por imóvel físico**, porque o mesmo apartamento
> aparece em até 7 anúncios; validação cruzada **GroupKFold(5)** só no treino, e o
> teste foi tocado **uma vez**. Imputação, padronização e one-hot moram **dentro do
> `Pipeline`**, reajustados em cada fold. Comparamos **seis modelos** sob protocolo
> pré-registrado: cada um declara a hipótese **antes** de rodar, e o critério de
> decisão foi escrito **em código, num script que nem abre as colunas de teste**.
> Vencedor: **Gradient Boosting ajustado**, com **vantagem declarada** — as cinco
> folds a favor e diferença média de 0,0563, onze vezes o nosso limiar. No teste:
> erro percentual mediano de **15,6%** e R² de 0,897 em log. E metade do que
> aprendemos não veio do modelo: veio de **três defeitos de dado** que a análise de
> resíduos expôs.

---

## 2. Quem fez o quê

Mapa reconstruído do histórico do git. **Confiram e corrijam os nomes antes de
apresentar** — o código ainda usa apelidos (`dev A`, `dev E`, `dev KNN`), e a
professora vai perguntar pelo nome real.

| Pessoa | Entregas | Arquivos / docs |
|---|---|---|
| **Vitor Reis** | Scraper do **chavesnamao** (sitemap, parser, sharding, retomada); candidatos **KNN** e **OLS**; métricas de treino+teste no `train.py` | `scraping/chaves_na_mao/`, `candidatos/knn.py`, `candidatos/ols.py`, `docs/scraping.md` |
| **João Victor Dantas** | Scraper do **ZapImóveis**; candidato **MLP** | `src/scrapping/zap_imoveis/` (branch), `candidatos/mlp.py`, `docs/modelos/mlp.md` |
| **Gabriel Ribeiro** | **Extração via LLM** (issue #9): descoberta empírica de atributos, schema dinâmico, lote, rotação de chaves, checkpoints; candidato **Árvore de Decisão** (issue #21) | `processing/extract_llm_features.py`, `candidatos/arvore.py`, `docs/issue_9_extracao_llm.md`, `docs/modelos/arvore.md` |
| **João Vitor Sampaio** | Reestruturação do repo; **etapas 3, 4, 4b–4e**: consolidação do one-hot, correlação, split/pipeline/CV, tuning, resíduos e permutação, bairros, repasse/ágio, `venda_direta`; **protocolo (#20)** e **comparação final (#25)**: critério em código, PCA, t-SNE | `features/`, `models/{dataset,train,tune,analysis,decisao,pca_variant}.py`, `docs/modelagem.md`, `docs/protocolo_comparacao.md`, `docs/comparacao_modelos.md` |
| **Micael Targino** | Limpeza do one-hot: dummies quase-constantes e sinônimos perdidos | `features/build_features.py` (PR #19) |

---

## 3. O fio da meada — o pipeline em 11 passos

Se a professora pedir "expliquem a metodologia", é **esta** a ordem.

```
 1. Coleta       2 portais, ~22,6 mil anúncios brutos
 2. Parsing      HTML → JSON estruturado
 3. Enriquec.    LLM (comodidades) + regex (campos numéricos da descrição)
 4. Dedup        chave (bairro, preço, área, quartos) → 15.476 imóveis únicos
 5. Features     231 colunas → 76: funde sinônimos, corrige numéricas, poda raras
 6. Correlação   ranking contra o alvo + poda de redundância (|r| ≥ 0,85)
 7. Split        80/20 AGRUPADO por imóvel físico, semente 42
 8. Pipeline     imputação + escala + one-hot DENTRO do fold
 9. Treino       GroupKFold(5) no treino; 6 candidatos + 2 referências
10. Decisão      critério pré-registrado, aplicado por script que não lê o teste
11. Diagnóstico  resíduos + permutação + PCA + t-SNE
```

**A frase que amarra tudo:** o passo 11 realimentou o passo 5 **três vezes**
(bairro, repasse, `venda_direta`). Não foi um pipeline linear — foi um ciclo.

---

## 4. Roteiro etapa por etapa

Formato: **o que dizer · o número · se perguntarem**.

### Etapa 1 — Coleta (Vitor Reis + João Victor)

**O que dizer.** Dois portais, dois scrapers independentes. O do chavesnamao lê a
lista de links do sitemap e visita anúncio a anúncio com sleeps de 3–7 s; é
**resumível** (salva a cada 25, escrita atômica) e **shardável** em 3 workers
disjuntos (~14 h → ~5 h). O do zap tem tracker de sessão e rate limiter próprios.

**Os números.** ~10,7 mil do chaves + ~11,8 mil do zap = **22.599 brutos**.

**Ética — decorem, isso cai.** Verificamos o `robots.txt`: páginas de anúncio são
permitidas. O `Content-Signal` do chavesnamao declara **`ai-train=no`**, então o
corpus **não** pode treinar nem fazer fine-tuning de LLM. `search=yes` e
`ai-input=yes` são permitidos, e modelagem tabular acadêmica também — que é
exatamente o nosso uso. Está no README e em `docs/scraping.md`.

**Se perguntarem:**
- *"Por que dois portais?"* → cobertura, e para poder **controlar artefato de
  portal**: o mesmo imóvel nos dois vira uma linha só, e quem só aparece em um fica
  marcado por `origem_anuncio`, que entra no modelo como variável de controle.
- *"Não é ilegal?"* → robots.txt respeitado, `ai-train=no` obedecido, sleeps
  educados (~0,5 req/s no total com 3 workers), sem redistribuição do conteúdo.

---

### Etapa 2 — Enriquecimento (Gabriel)

Duas técnicas diferentes, e a distinção importa.

**(a) LLM sobre a descrição — comodidades.** Metodologia em **duas etapas**, para
não escolher os atributos no chute:
1. **Amostragem aberta** em 1.000 imóveis — a LLM extrai *tudo* que o texto cita,
   sem schema fixo → ranking empírico de frequência;
2. **Schema dinâmico** com os **45 atributos** mais frequentes + campo
   `diferenciais_unicos`, para não perder o exótico.

Infra: lote de 6 anúncios por chamada, rotação round-robin de chaves com failover,
checkpoint atômico. Cobertura: **6.966 imóveis** com descrição rica, 100% do escopo
do chavesnamao.

**(b) Regex sobre a descrição — campos numéricos** (`quartos`, `suites`,
`banheiros`, `garagens`). **Por que regra e não LLM aqui:** determinístico (o mesmo
texto sempre dá o mesmo resultado, ao contrário de uma chamada com temperatura),
testável em CI sem chave de API, roda em segundos.

**A validação que autoriza usar isso — ponto forte da etapa.** Antes de preencher
qualquer célula ausente, medimos a precisão **onde o campo estruturado já
existia**: garagens 94,4%, suítes 93,6%, quartos 90,4%, banheiros 86,0% de acerto
exato. Sem essa medida, o preenchimento seria chute com cara de dado. Resultado:
`suites` deixou de faltar em **54,3%** da base e passou a faltar em **46,9%**.

**E o que decidimos NÃO extrair.** `area_util` — nenhum padrão passou de **67%**,
porque as descrições citam várias áreas (privativa, total, lazer, terreno). A
diferença mediana é zero, logo não é viés corrigível, é dispersão. `area_util` é a
2ª feature mais importante do modelo; injetar ~35% de erro nela para preencher 6,5%
de ausência sai **pior** que manter o nulo, que o `SimpleImputer` já trata.

**Se perguntarem:**
- *"Não vaza o alvo usar LLM?"* → não: cada linha é lida **isoladamente**, sem
  estatística agregada da base, então pode rodar antes do split. O que vaza é
  agregação — e foi por isso que removemos `bairro_preco_m2_medio`.
- *"A LLM cobriu tudo?"* → não, e está documentado: no zap **78% dos anúncios não
  têm descrição** utilizável. Rodar a extração completa lá alcançaria ~2,6 mil
  anúncios (16% da base) e não resolveria o artefato de portal — por isso
  `origem_anuncio` fica no modelo como controle.
- *"Por que reverteram a extração parcial do zap?"* → porque ela **não era amostra
  aleatória**: a cota da API acabou no meio da fila, e a fila é a ordem de ranking
  de busca do portal. Processados tinham mediana de R$ 498 mil contra R$ 430 mil
  dos não processados, **Mann-Whitney p = 0,0021**. "Ter atributo extraído" viraria
  proxy de "é caro". E não havia ganho: CV 0,2232 sem, 0,2238 com.

---

### Etapa 3 — Features e correlação (João Vitor Sampaio + Micael)

**O problema de partida.** O one-hot já existia — gerado **três vezes** por
caminhos que nunca conversaram (LLM sobre a descrição, HTML do chaves, HTML do
zap). 231 colunas com seis defeitos:

1. `NaN` não significava "não tem", significava "o outro portal";
2. colunas duplicadas (`deposito`/`depósito`, `gas_canalizado`/`gás_encanado`);
3. **colinearidade perfeita**: `piscina = piscina_LLM OR comodidade_piscina`;
4. `suites` e `banheiros` **numéricos sobrescritos por `True`/`False`** — 7.006 e
   8.988 células;
5. numéricos como texto (`'625.000'`, `'--'`, `'Isento'`);
6. 105 binárias em menos de 1% dos imóveis + pseudo-atributos (`apartamento`,
   `lazer`, `conforto`).

**O que `build_features` faz.** Reconstrói as numéricas destruídas a partir do JSON
bruto casando por URL (recuperou 6.911 de `suites` e 8.826 de `banheiros`); funde
binárias equivalentes por **OR** tratando `NaN` como ausência (205 → 157); descarta
pseudo-atributos e comodidades com < 1%; anula valores fora da faixa plausível
(havia preço de R$ 470 milhões e área de 58 milhões de m²). **231 → 76 atributos.**

**As duas features removidas por vazamento — decorem, é a pergunta clássica.**
- `bairro_preco_m2_medio` parecia fonte externa (vinha de um CSV), mas comparada
  com a mediana de preço/m² calculada **da própria base** deu correlação **0,996**,
  erro relativo mediano 2,4%. É agregação do alvo disfarçada. Era a 6ª feature mais
  forte do ranking. Removida.
- `anunciante_qtd_anuncios` era `value_counts()` sobre a base inteira — a contagem
  incluía as linhas que virariam teste.
- **O critério que ficou:** só entra na matriz transformação determinística ou com
  constante fixa; qualquer coisa que **aprenda estatística dos dados** acontece
  depois do split, dentro do `Pipeline`.
- **O que não foi removido, e por quê:** os limites de plausibilidade são constantes
  de domínio escritas à mão, não estatística da base — isso é saneamento, não
  aprendizado. Se um dia virarem IQR ou z-score, migram para dentro do `Pipeline`.

**Correlação.** Spearman e Pearson contra `preco_venda`, `log_preco` e `preco_m2`.
Binária × contínua é ponto-bisserial e binária × binária é phi — as duas são a
fórmula de Pearson, então **uma matriz só** cobre os três casos. Poda de
redundância: em cada par com |r| ≥ 0,85 cai o menos correlacionado com o alvo — só
**1 par** sobreviveu (`area_total` × `area_util`, r = 0,92), justamente porque a
colinearidade pesada já tinha morrido na consolidação.

Top do ranking: `area_util` +0,696 · `suites` +0,658 · `area_total` +0,630 ·
`garagens` +0,625 · `banheiros` +0,533 · `quartos` +0,503.

**Se perguntarem:** *"`features_selecionadas.csv` é usado no treino?"* → **não**, de
propósito: a seleção usa correlação com o alvo de **todas** as linhas. É artefato de
EDA e relatório; usá-lo como filtro seria vazamento.

---

### Etapa 4 — Split, pipeline e treino (João Vitor Sampaio)

**Alvo.** `log(preco_venda)`. Em reais, assimetria **5,92** e curtose **80,4**;
mediana R$ 580 mil, máximo R$ 19,8 milhões. Em log a assimetria cai para **−0,32**.
As métricas voltam para reais na avaliação, porque é nelas que a resposta faz
sentido.

**Split — o ponto mais importante da etapa.** `GroupShuffleSplit` 80/20,
`random_state=42`, **agrupado por imóvel físico**. A base tem o mesmo apartamento
anunciado várias vezes (1.050 grupos somando 2.328 anúncios, o maior com **7
cópias**), porque a dedup entre portais não pega repetição dentro do mesmo portal.
Com split aleatório o mesmo imóvel cairia dos dois lados e a métrica mediria
**memorização**. A assinatura é `(preço arredondado ao milhar, área, quartos,
banheiros, garagens)`; anúncio sem preço ou área vira grupo próprio, para não se
juntar num grupo gigante de nulos. Resultado: **12.214 treino / 3.087 teste, 0
grupos dos dois lados** — e o `train.py` **aborta** se não for zero.

**Pré-processamento, tudo dentro do `Pipeline`:**

```
ColumnTransformer
├── numéricas (8):  SimpleImputer(mediana, add_indicator) → StandardScaler
├── binárias (63):  passthrough  (ou StandardScaler, se o candidato pedir)
└── nominais (5):   OneHotEncoder(min_frequency=30,
                                  handle_unknown='infrequent_if_exist')
```

- **`add_indicator=True`**: `iptu` falta em 80% dos anúncios, `area_total` em 61%,
  `suites`/`condominio` perto de 47% e 54%. O **silêncio do anunciante é
  informação**; imputar sem a indicadora apagaria o sinal.
- **Mediana, não média**: mesmo motivo do log — caudas pesadas.
- **`min_frequency=30`**: categoria rara vai para um balde comum, contado **dentro
  do fold**. `handle_unknown='infrequent_if_exist'` manda bairro inédito no teste
  para esse mesmo balde, em vez de quebrar.

**O vazamento estrutural — história obrigatória.** Antes, o one-hot era
`pd.get_dummies` sobre a base inteira, com dois cortes: bairro com < 30 imóveis
virava `outros`, e dummy abaixo de 1% era descartada. Nenhum dos dois olha o alvo —
então **não é vazamento de alvo, é vazamento estrutural**: o *conjunto de colunas*
era definido usando as linhas que virariam teste. Previmos que corrigir custaria
acurácia; **aconteceu o contrário** (GB 0,2232 → 0,2155, Ridge 0,3037 → 0,2906),
porque os cortes globais **destruíam informação**.

**Validação.** `GroupKFold(5)` **sobre o treino**, com os mesmos grupos. Se fosse
`KFold` simples, as duplicatas voltariam a atravessar os folds e a CV herdaria o
problema que o split resolveu. **O teste é tocado uma única vez, no fim.**

**Hiperparâmetros (`tune.py`).** `GridSearchCV` com o **mesmo** `GroupKFold` e o
**mesmo** conjunto de treino. Três lições:
1. **Ótimo na borda = grade mal-posta.** `max_leaf_nodes` melhorava
   monotonicamente até o teto (15→0,2346, 31→0,2291, 63→0,2262); estendemos até 255
   e a curva virou (127 é o ótimo interior). `min_samples_leaf` caiu na borda **três
   vezes**, uma a cada mudança de base. Lição: **mudou a base, remede as bordas**.
2. **Eixo morto.** `l2_regularization` deu 0,2234 com 0 e 0,2235 com 1 — diferença
   na quarta casa. Fixado no default, liberando metade das configurações da grade.
3. **A CV escolhe, o teste relata.** Escolher a melhor de 32 configurações torna
   aquele score otimista: numa passada só **4%** do ganho da CV apareceu no teste;
   noutra, **100%**. Não voltamos atrás depois de ver o teste — seria usar o teste
   para selecionar.

---

### Etapa 4b — Resíduos e importância por permutação (João Vitor Sampaio)

Tudo medido **no teste**, sobre o `gradient_boosting_ajustado`. Importância no
treino responde "do que o modelo se lembrou"; a pergunta do projeto é "do que ele
precisa para acertar num imóvel que nunca viu". O preço é que o teste já foi usado
para relatar a métrica final — por isso **nada daqui volta como seleção de
atributo**. É leitura, não decisão.

**Três achados:**

1. **O modelo puxa tudo para o meio.** Viés vai de **+7,9% no Q1** a **−14,3% no
   Q5**, trocando de sinal monotonicamente — regressão à média clássica de um
   estimador que minimiza erro. *Detalhe que impressiona:* no gráfico clássico
   (resíduo × **previsto**) a linha é reta em zero, sem viés nenhum; o viés só
   aparece condicionando no **real**. Os dois estão certos e medem coisas
   diferentes — diagnosticar só pelo gráfico padrão esconderia o efeito.
2. **`bairro` e `area_util` valem 0,42 dos 0,61 de importância total.** Embaralhar
   `bairro` piora o MAE em 0,216 — mais que o MAE final inteiro (0,200). **34 dos 76
   atributos têm importância indistinguível de zero**, quase todos comodidades:
   dado bairro e área, comodidade não acrescenta. *Nota técnica:* permutamos o
   **atributo inteiro**, não dummy a dummy — permutar uma dummy de bairro por vez
   deixaria as outras entregando a resposta e a importância sairia zero por
   construção.
3. **Correlação ≠ importância.** `com_closet` está no top-10 de correlação e vale
   **zero** para o modelo (closet é proxy de apartamento grande em bairro caro).
   `bairro_bessa` tem correlação **0,010** e é uma das dummies mais úteis: o Bessa
   tem preço mediano colado na mediana geral, então a correlação linear é nula por
   construção; o que ele informa é o **preço por m²** dentro dele, que só existe em
   interação com `area_util`. **É esta a resposta prática de por que o boosting
   ganha do Ridge.**

---

### Etapa 4c — Canonização dos bairros (João Vitor Sampaio)

**14,0% dos anúncios tinham o bairro errado** — no atributo mais importante do
modelo. Três defeitos:

1. **Casamento por substring na ordem da lista.** `"cabo branco"` vinha antes de
   `"altiplano"`, então **511 anúncios do Altiplano Cabo Branco viravam Cabo
   Branco** — bairros com 26% de diferença de preço/m² fundidos numa categoria só.
   É o pior erro possível: não embaralha, **junta duas faixas de preço sob o mesmo
   rótulo**.
2. **Um fallback que inventava bairros:** endereço sem match virava a primeira
   palavra com mais de 3 letras → `avenida`, `doutor`, `telegrafista`. 1.589
   anúncios (9,8%) em categorias que não existem. O balde `avenida` tinha IQR de
   preço/m² de R$ 6.898 contra R$ 1.316 de um bairro real — agrupava a cidade toda.
3. Nomes não-canônicos (`geisel`, `planalto`, `valentina`).

**A correção.** O endereço tem estrutura, e os portais escrevem diferente (vírgula
no chaves, hífen no zap). A nova função quebra por vírgula **e** hífen, percorre do
fim para o começo (o bairro fica antes da cidade) e casa contra os **64 bairros
oficiais** por conjunto de tokens, com o **nome mais específico vencendo**. **Nunca
inventa:** o que não casa vira `nao_informado` — sobraram 13 anúncios (0,08%).

**329 valores distintos → 66.** Efeito colateral: como `bairro` entra na chave de
dedup, apareceram **579 duplicatas** que antes escapavam; a base cai de 16.162 para
15.583 **sem perder imóvel**. E a matriz pós-one-hot cai de **349 para 131 colunas**
com o modelo **melhorando**.

**A medição honesta.** Comparar as duas execuções não isola nada, porque a base
mudou junto. Fizemos **A/B sobre as mesmas linhas e as mesmas folds**, trocando só a
coluna: **0,2144 → 0,2097**, com os **5 de 5 folds** favorecendo o canônico.

**Uma previsão que não se confirmou.** Prevíamos que `cabo_branco` melhoraria,
"porque hoje são dois bairros somados". Piorou: 16,4% → 17,7%. Motivo: depois de
puro, Cabo Branco tem o **maior IQR de preço/m² de todos** (R$ 7.882) — é a orla de
alto padrão, mesmo bairro com imóveis radicalmente diferentes. Misturar o Altiplano,
mais homogêneo, **diluía** a dificuldade. Purificar uma categoria não a torna mais
fácil — torna o número honesto.

---

### Etapa 4d — O preço que não era preço (João Vitor Sampaio)

Os anúncios mais baratos não eram imóveis baratos: eram **repasses de
financiamento**. O valor da etiqueta é o ágio pago pelas chaves, e o comprador ainda
assume as parcelas. *"Repasse no Valentina: Chaves R$ 21.500 e Parcela Menor que
Aluguel (R$ 719)"*. **Dois produtos com o mesmo rótulo `preco_venda`.**

**Como escolhemos o piso.** A distribuição de preço/m² **não tem vale** — é contínua
de R$ 250 a R$ 4.000. Então o piso é escolha, não descoberta, e foi calibrado contra
um **sinal independente**: a palavra "repasse"/"ágio" no texto, que 177 anúncios
declaram.

| piso | descartados | declaram repasse | precisão |
|---|---|---|---|
| R$ 1.000 | 111 | 84 | **76%** |
| R$ 1.500 | 210 | 104 | 50% |

A precisão desaba porque entre R$ 1.000 e R$ 1.500 entra uma população **legítima**:
311 vendas diretas/leilão da Caixa, com preço/m² entre R$ 1.164 e R$ 1.899. O piso
fica **abaixo** delas de propósito, ao custo declarado de 21 falsos positivos. **O
texto calibrou o piso; não filtra nada** — usar "repasse" como filtro descartaria
anúncio comum que cita a palavra no rodapé da imobiliária.

**O defeito inverso.** Nove anúncios têm a **área** errada e o preço certo —
separador decimal perdido: 988 m² num anúncio intitulado *"98m²"*. Aqui **anula-se a
área, não a linha**, e a regra roda **antes** do piso, senão os R$ 550.000 de Tambaú
sairiam junto. Um teto absoluto de área mataria as coberturas reais de R$ 11 e
R$ 19,8 milhões — por isso a regra olha o **par** (área, preço/m²).

**A honestidade que vale ponto.** A base foi de 15.583 → 15.476 e a CV de 0,2057 →
0,1998. **Esses números não são comparáveis** — mudaram a base *e* o teste. O A/B
sobre as mesmas 3.087 linhas mostra efeito real de **+0,0031**, abaixo do limiar de
0,005. **A afirmação correta não é "o modelo melhorou": é que a base ficou certa.**

---

### Etapa 4e — A binária `venda_direta` (João Vitor Sampaio)

262 anúncios de venda direta/leilão de banco, com preço/m² mediano de **R$ 1.665
contra R$ 9.089** do resto — 18% do mercado. Termos calibrados um a um:
`caixa economica` e `aceita fgts` foram **reprovados** (opção de financiamento em
anúncio comum, preço/m² de mercado); `matricula` foi reprovado por **redundância** —
254 dos 259 já batiam em `venda direta`, e os 5 que ele adiciona sozinho custam de
R$ 2.790 a R$ 8.694/m². *Um termo pode parecer bom só por concordar com quem já
estava certo.*

**Três resultados que a professora vai gostar:**

1. **Previsão registrada antes: 0,005 a 0,015. Medido: 0,0010** — errada por uma
   ordem de magnitude. O erro tem explicação: o **Ridge ganha 0,0045, 4,5× mais que
   o boosting**, porque a árvore já reconstruía "apartamento pequeno *naquele*
   bairro" a partir de `bairro` × `area_util`. Previmos o ganho do modelo que não
   tinha como já saber.
2. **Permutação ≠ ablação.** `venda_direta` é a **6ª mais importante** por
   permutação e removê-la custa só 0,0010. Permutação quebra a coluna com o modelo
   já treinado esperando por ela → mede **dependência**. Ablação treina de novo sem
   ela → mede **insubstituibilidade**. A diferença entre as duas **é** a
   redundância. Confirmação: a importância de `bairro` caiu de 0,2394 para 0,2160
   quando `venda_direta` entrou. **Importância alta não justifica manter feature.**
3. **77% do ganho vem de quem não é leilão.** No segmento o efeito por anúncio é 19×
   maior, mas o resto é 65× mais numeroso: 47 × 0,0428 = 2,01 contra
   3.040 × 0,0022 = 6,69. Marcar o leilão serve menos para acertar o leilão e mais
   para **parar de contaminar o resto**.

---

### Etapa 5 — Protocolo e comparação final (issue #25)

#### 5.1 A infraestrutura (João Vitor Sampaio)

`models/candidatos/` é um **registro por descoberta**: cada dev cria um arquivo e
exporta uma constante `CANDIDATO`; não existe lista central para editar. É por isso
que cinco pessoas trabalharam na mesma semana sem colidir — quando duas mexeram em
`build_features.py` na mesma janela, 4 dos 5 conflitos foram em arquivo gerado.

**As quatro regras do protocolo:**
1. Mesmo split, mesmas folds, mesmo `Pipeline` (`SEMENTE = 42`). Preparação própria
   invalida a comparação.
2. Critério de decisão declarado antes dos resultados.
3. O teste é tocado **uma vez, por uma pessoa** — cinco pessoas olhando o teste
   durante o desenvolvimento o transformam num segundo conjunto de validação.
4. **Hipótese registrada antes de rodar.**

**Por que a regra 4 virou código.** O `dataclass` **recusa** hipótese vazia e o teste
recusa hipótese com menos de 40 caracteres. Não é burocracia: três dos achados mais
úteis do projeto foram previsões **erradas**. Escrever a hipótese depois transforma
qualquer resultado em confirmação.

**`escalar_binarias` — cada candidato declara o que precisa.** As contínuas saem do
`StandardScaler` com variância 1; uma binária com *p* = 0,1 tem variância 0,09. Para
**KNN e MLP** isso faz as contínuas dominarem a distância e o gradiente; para
**árvore e Ridge** é irrelevante — a árvore corta por limiar, e limiar em 0/1 não
muda com a escala. **Por que `StandardScaler` e não `MinMaxScaler`:** MinMax em dado
que já é 0/1 é literalmente a identidade; só a padronização iguala as variâncias.

#### 5.2 O critério de decisão — escrito em código, antes do resultado

Implementado em `src/imoveis_jp/models/decisao.py`:

1. Vence o **menor MAE médio do `GroupKFold(5)`** no treino, sobre log(preço).
2. A vantagem só é **declarada** se a diferença pareada favorecer o mesmo modelo
   nas **cinco folds** **e** a diferença média for **≥ 0,005**.
3. Se qualquer condição falhar: **empate técnico**, desempatado nesta ordem —
   **explicabilidade → custo de previsão → número de hiperparâmetros** (as duas
   primeiras ordens estão tabeladas no próprio módulo).
4. O teste é avaliado **depois**, uma vez, e não participa da decisão.

**A garantia é estrutural, não moral:** `decisao.py` **nunca abre uma coluna
`*_teste`** — por construção. E, a partir desta issue, a CV é gravada **fold a
fold** (`cv_mae_por_fold.csv`); antes só a média era registrada, e só a média não
diz se a vantagem é consistente ou se veio de uma fold sortuda.

**Por que 0,005:** o desvio entre folds das melhores configurações do projeto fica
entre 0,0032 e 0,0043. Diferença menor que um desvio não se sustenta — já aconteceu
de a busca de hiperparâmetros eleger uma configuração por 0,0005 num eixo que não
tinha efeito nenhum.

#### 5.3 O resultado

Ranking por CV: **gradient_boosting_ajustado → ridge → ols → arvore_decisao → mlp →
knn.** Comparação pareada entre 1º e 2º, fold a fold: **+0,0582 · +0,0560 · +0,0517
· +0,0569 · +0,0586**. Cinco de cinco a favor, média **0,0563** — **11× o limiar**.

> **RESULTADO: VANTAGEM DECLARADA. Vencedor: `gradient_boosting_ajustado`.**

**Tabela oficial** (uma única rodada de `train.py`, ordenada pela **CV**, não pelo
teste, para a ordem não parecer escolhida pelo teste):

| modelo | Hipótese registrada | CV MAE(log) | Teste MAE (R$) | Erro % mediano | R² (log) |
|---|---|---|---|---|---|
| **gradient_boosting_ajustado** | "≈ 0,20; ganha por interação e não-linearidade" | **0,1988 ± 0,0030** | **R$ 170.150** | **15,6%** | **0,897** |
| gradient_boosting (padrão) | referência, não candidato | 0,2076 ± 0,0056 | R$ 172.892 | 16,1% | 0,892 |
| ridge | "0,25–0,27; a margem mede quanta interação existe" | 0,2551 ± 0,0022 | R$ 249.200 | 19,1% | 0,844 |
| ols | "≈ Ridge; o gargalo é forma funcional, não regularização" | 0,2551 ± 0,0022 | R$ 249.118 | 19,1% | 0,844 |
| arvore_decisao | "overfitting severo sem poda; perde do boosting" | 0,2846 ± 0,0040 | R$ 244.051 | 20,0% | 0,780 |
| mlp | "0,21–0,23; supera Ridge, perde do boosting" | 0,3162 ± 0,0081 | R$ 286.047 | 20,8% | 0,746 |
| knn | "entre Ridge e boosting; se ficar pior que Ridge, é a dimensionalidade" | 0,3202 ± 0,0064 | R$ 269.182 | 23,8% | 0,715 |
| baseline_mediana | referência, piso de sanidade | 0,6183 | R$ 443.793 | 43,1% | −0,002 |

**As cinco leituras que amarram a comparação:**

- **OLS ≡ Ridge até a quinta casa** (0,25513 vs 0,25512). Isso, somado à busca de
  `alpha` que devolveu o próprio default de 0,1 a 1.000, prova que o gargalo do
  linear **não é regularização, é forma funcional**. E descarta a hipótese
  alternativa do OLS: a colinearidade do one-hot **não** estava inflando os
  coeficientes — o `min_frequency=30` já agrupa as categorias raras antes do
  regressor.
- **O ajuste de hiperparâmetros vale pouco perto da escolha de família.** O tuning
  rendeu 0,0088 sobre o GB padrão (4,2%), contra 0,056 de distância para o linear.
  **A família do modelo importa mais que o tuning.**
- **A árvore isolada perde na CV (0,2846) mas ganha do Ridge/OLS no MAE em reais do
  teste** (R$ 244 mil vs R$ 249 mil). Isso é **sinal de variância alta**, não de que
  a CV escolheu errado — a árvore memoriza o treino (MAE de treino: R$ 2.153) e o
  critério nunca comparou árvore contra Ridge por essa métrica. **Atenção:** o 0,2846
  é a árvore **sem poda**, que é a configuração inscrita; a podada chega a 0,2450 e
  ficaria à frente dos lineares — ver §6.5.
- **KNN e MLP perdem até do linear.** São exatamente os dois modelos que dependem de
  geometria (distância e gradiente) numa matriz de 132 colunas majoritariamente
  binárias e esparsas. A hipótese do KNN **previu esse cenário e nomeou a causa
  antes de rodar** — maldição da dimensionalidade. Hipótese refutada na direção,
  confirmada no mecanismo.
- **MLP e KNN têm os maiores desvios entre folds** (0,0081 e 0,0064, contra 0,0022 do
  Ridge) — os dois modelos que dependem de geometria no espaço de atributos são também
  os mais sensíveis a quais imóveis caem em cada fold.

#### 5.4 Variante PCA — a hipótese registrada era "piora", e piorou nos seis

**Por que prevíamos piora:** PCA é uma projeção **linear**, e o que falta ao modelo
linear é justamente a interação **não-linear** área × bairro — rotacionar o espaço
não cria interação, só reduz dimensão. Para as árvores o argumento é outro: um
componente principal (combinação linear de dezenas de dummies de bairro) não tem
limiar interpretável nem alinhado aos cortes que a árvore faria. E destrói a
interpretabilidade que dá o resultado mais forte do projeto.

PCA com **95% da variância retida**, mesmo split e mesmas folds:

| modelo | sem PCA | com PCA | Δ | componentes |
|---|---|---|---|---|
| gradient_boosting_ajustado | 0,1988 | 0,2413 | +0,0425 | 50 |
| ridge | 0,2551 | 0,3240 | +0,0688 | 50 |
| ols | 0,2551 | 0,3240 | +0,0688 | 50 |
| arvore_decisao | 0,2846 | 0,3726 | +0,0880 | 50 |
| mlp | 0,3162 | 0,4077 | +0,0915 | 69 |
| knn | 0,3202 | 0,3341 | +0,0139 | 69 |

**Confirmada nos seis, sem exceção.** O detalhe que rende: com PCA, o **melhor
modelo do projeto (0,2413) fica pior que o pior modelo linear sem PCA (0,2551)** —
50 componentes lineares invertem o ranking inteiro. E quem menos sofre é o KNN
(+0,0139), o que faz sentido: a distância euclidiana dele já estava tão diluída pela
alta dimensionalidade que reduzir de 132 para 69 quase não muda a geometria que
importa — ao contrário dos modelos que dependiam de cortes ou coeficientes sobre
colunas **interpretáveis**.

#### 5.5 t-SNE — EDA pura

Projeção 2D sobre as 15.301 linhas, colorida por quintil de preço
(`docs/figuras/tsne_precos.png`). **Não entra em nenhum `Pipeline`, e o motivo é
técnico:** t-SNE não tem `.transform` para dado novo — cada chamada reprojeta o
conjunto inteiro, então não serve como passo de um pipeline que precisa prever fora
da amostra. Rodamos sobre a base inteira porque é leitura exploratória, não
avaliação de modelo.

**O que se vê:** dezenas de aglomerados pequenos e bem definidos — esperado, porque
com 132 colunas majoritariamente binárias o t-SNE agrupa por **coincidência exata de
padrão categórico** antes de qualquer coisa contínua. Dentro da maioria dos
aglomerados as faixas de preço aparecem **misturadas** — o que é coerente com o
argumento central do projeto: dois imóveis do mesmo bairro têm preços bem diferentes
se a área mudar, porque o preço depende da **combinação** área × bairro. Os poucos
aglomerados quase puramente Q1 são bairros de padrão uniformemente baixo, onde a
área pesa menos. **Leitura visual, não conclusão estatística — não influenciou
nenhum modelo.**

#### 5.6 O viés que continua em aberto — diga isso antes que perguntem

`venda_direta` resolveu um caso: hoje é coluna explícita, disponível igualmente para
todos os modelos. Mas a interação **`bairro × area_util` ainda não é coluna
explícita** na matriz. Boosting e árvore a reconstroem sozinhos, por construção;
Ridge, OLS, KNN e MLP não têm como. **Parte da distância medida entre a família de
árvores e o resto é engenharia de atributos que falta, não capacidade do
algoritmo.** Está registrado como limitação conhecida na §8 de
`docs/comparacao_modelos.md` — não como algo resolvido.

---

## 5. Cartão de bolso — números para saber de cor

| | |
|---|---|
| Anúncios brutos coletados | 22.599 (2 portais) |
| Base final | **15.476** linhas · **76** atributos |
| Com preço (usados no modelo) | 15.301 · **14.022** imóveis físicos |
| Split | **80/20 agrupado**, semente **42** → 12.214 treino / 3.087 teste / **0** vazados |
| Colunas depois do one-hot | **132** |
| Alvo | `log(preco_venda)` — assimetria 5,92 → −0,32 |
| Validação | **GroupKFold(5)** no treino, gravada **fold a fold**; teste tocado **1 vez** |
| Critério | menor MAE de CV + 5/5 folds a favor + Δ ≥ **0,005** |
| **Vencedor** | **Gradient Boosting ajustado — vantagem declarada** (Δ = 0,0563) |
| CV do vencedor | **0,1988 ± 0,0030** |
| Teste do vencedor | MAE **R$ 170.150** · erro mediano **15,6%** · R² (log) **0,897** |
| Treino do vencedor | MAE(log) 0,137 · R² 0,948 · erro mediano 10,4% |
| Baseline | erro mediano 43,1% · R² −0,002 |
| Top-2 features | `bairro` +0,216 · `area_util` +0,206 (0,42 de 0,61 do total) |
| Atributos com importância ≈ 0 | **34 de 76** |
| PCA | piora **os seis**, de +0,014 (KNN) a +0,092 (MLP) |

**Hiperparâmetros vencedores:** `learning_rate=0.05`, `max_iter=500`,
`max_leaf_nodes=127`, `min_samples_leaf=5`, `l2_regularization=0.0`.

---

## 6. Banco de perguntas cruzadas — "pergunta da tarefa X pra quem fez Y"

**Leia como se a pergunta fosse pra você.**

### 6.1 Perguntaram de COLETA e você não fez o scraper

- *"Como garantiram que não pegaram o mesmo imóvel duas vezes?"* → Dedup entre
  portais pela chave `(bairro, preço, área, quartos)`: 22.599 brutos − 7.016
  duplicatas = 15.583, depois 15.476 com a limpeza de repasses. **Dentro do mesmo
  portal a dedup não pega** — por isso o split é agrupado por assinatura física.
- *"Quanto tempo levou?"* → ~8–10 s por anúncio com sleeps educados; ~24–30 h em
  série, ~5 h com 3 workers em fatias disjuntas. Resumível, salva a cada 25 com
  escrita atômica.
- *"E a ética?"* → robots.txt verificado, `ai-train=no` respeitado, ~0,5 req/s.

### 6.2 Perguntaram da LLM e você não fez a extração

- *"Por que LLM e não regra?"* → Usamos **os dois**, para coisas diferentes: LLM
  para **comodidades** em texto livre, onde o vocabulário é aberto (e mesmo lá o
  schema foi descoberto empiricamente numa amostra de 1.000 anúncios); **regex**
  para os **campos numéricos**, porque é determinístico e testável em CI.
- *"Como sabem que a extração está certa?"* → Medimos a precisão onde o campo
  estruturado **já existia**: 94,4% garagens, 93,6% suítes, 90,4% quartos, 86,0%
  banheiros. E rejeitamos `area_util` porque nenhum padrão passou de 67%.
- *"A LLM não pode ter vazado o preço?"* → Ela lê **uma descrição por vez**, sem
  acesso ao alvo agregado nem a outras linhas. O que vaza é **agregação** — e foi
  por isso que removemos `bairro_preco_m2_medio`.

### 6.3 Perguntaram de FEATURES/CORRELAÇÃO e você fez modelo

- *"Por que 231 colunas viraram 76?"* → Porque o one-hot vinha de três geradores
  independentes que produziam **a mesma informação com nomes diferentes**
  (colinearidade perfeita), `NaN` significando "o outro portal", e 105 binárias
  presentes em menos de 1% dos imóveis.
- *"Fizeram seleção de features?"* → Geramos ranking e poda de redundância
  (|r| ≥ 0,85) **como EDA**, mas **não** filtramos o treino por eles — a seleção usa
  correlação com o alvo de todas as linhas, incluindo o teste. Quem faz seleção real
  é o modelo, e a permutação mostra que 34 dos 76 atributos são dispensáveis.
- *"E PCA, não reduziria a dimensão?"* → Testamos, com hipótese registrada antes:
  **piora os seis modelos**, de +0,014 a +0,092. É projeção linear, e o que falta é
  interação não-linear; além disso destrói a interpretabilidade que dá o resultado
  mais forte do projeto. Está na §5 de `docs/comparacao_modelos.md`.

### 6.4 Perguntaram de SPLIT/VAZAMENTO e você fez scraping ou LLM

Estas três **todo mundo** precisa saber:

- *"Por que split agrupado?"* → O mesmo apartamento aparece em até **7 anúncios**
  (1.050 grupos, 2.328 anúncios). Com split aleatório ele cairia dos dois lados e a
  métrica mediria memorização. O `train.py` verifica e **aborta** se algum grupo
  aparecer nos dois.
- *"Onde acontece o pré-processamento?"* → **Dentro do `Pipeline` do sklearn** —
  imputação, padronização e one-hot. O `fit` é refeito em cada fold e nenhuma
  estatística do fold de validação entra no de treino.
- *"Teve vazamento em algum momento?"* → Teve, e achamos numa auditoria: (a) o
  one-hot rodava sobre a base inteira — **vazamento estrutural**, o conjunto de
  colunas era definido usando o teste; (b) `bairro_preco_m2_medio` era agregação do
  alvo disfarçada de fonte externa (correlação 0,996 com a mediana da própria base);
  (c) `anunciante_qtd_anuncios` era `value_counts` da base inteira. Os três
  corrigidos — e corrigir (a) **melhorou** o modelo.

### 6.5 Perguntaram do SEU modelo, comparando com o dos outros

- *(KNN)* *"Por que seu modelo é o pior?"* → 132 colunas majoritariamente binárias e
  esparsas: a distância euclidiana perde poder de discriminação e "vizinho mais
  próximo" passa a significar "imóvel do mesmo bairro, independente do resto". **A
  hipótese previu esse cenário antes de rodar** e nomeou a causa — maldição da
  dimensionalidade. É hipótese refutada na direção e confirmada no mecanismo.
- *(KNN)* *"Seu erro de treino é praticamente zero. Não é overfitting?"* → É **por
  construção**: com `weights="distance"`, o vizinho a distância zero (o próprio
  ponto) recebe peso infinito, então o modelo reproduz o treino exatamente. O que
  vale é a CV (0,3202) e o teste (0,3282). É exatamente por isso que **CV e teste
  existem** e o erro de treino não decide nada.
- *(KNN)* *"Ele tem alguma desvantagem além do erro?"* → Sim, estrutural: é o único
  dos seis cujo **custo de previsão cresce com o tamanho da base** — precisa varrer
  ou indexar o treino inteiro a cada previsão. Isso está tabelado em `decisao.py` e
  pesaria contra ele em caso de empate técnico.
- *(MLP)* *"Por que a rede neural perdeu para uma regressão linear?"* → Dado
  tabular de médio porte com 63 binárias esparsas é o terreno onde ensembles de
  árvore dominam: cortes ortogonais se ajustam melhor a atributos discretos, enquanto
  o gradiente estocástico converge com ruído alto nessa matriz. Prevíamos 0,21–0,23
  e veio 0,3162 — **erramos, e o registro prévio da hipótese é o que torna isso um
  resultado, não uma desculpa.**
- *(MLP)* *"Vocês usaram early stopping?"* → **Não, de propósito.** O
  `MLPRegressor` com `early_stopping=True` separa 10% do treino por **amostragem
  aleatória simples**, o que fura o agrupamento por imóvel físico: cópias do mesmo
  apartamento cairiam na validação interna e camuflariam o sobreajuste. A
  generalização é avaliada só pelo `GroupKFold(5)`. (Mesmo defeito existe no
  boosting, e está declarado como limitação — lá o `early_stopping='auto'` liga
  sozinho acima de 10.000 amostras.)
- *(Árvore)* *"Sua árvore não é só um Gradient Boosting pior?"* → Ela existe por
  **explicabilidade** — é a nº 1 nessa dimensão no critério de desempate — e mostra
  a curva de overfitting de forma didática: sem poda, MAE de treino de **R$ 2.153**
  contra CV de 0,2846. Podada por `ccp_alpha=5e-5` e `min_samples_leaf=5`, a CV cai
  para 0,2450 e o MAE de teste para R$ 200.983. E ela é a **ponte conceitual** para
  o boosting, que é um ensemble de centenas de árvores rasas corrigindo resíduo.
- *(Árvore)* *"Por que na tabela final ela aparece com 0,2846 e não 0,2450?"* →
  Porque o candidato **inscrito** é o `DecisionTreeRegressor` sem poda, e isso é
  deliberado: é ele que produz a curva de overfitting, que é o entregável didático da
  issue. O 0,2450 é a configuração vencedora do `GridSearchCV` (`ccp_alpha=5e-5`,
  `min_samples_leaf=5`), medida e documentada em `docs/modelos/arvore.md` §4. **A
  leitura correta da tabela oficial é "cada modelo como foi inscrito", não "cada
  modelo no seu melhor"** — e isso vale também para KNN e MLP, que estão na
  configuração inicial dos donos. Só `ridge` e `gradient_boosting_ajustado` trazem a
  vencedora de busca.
- *(Árvore)* *"Então a comparação é injusta com a árvore?"* → É uma assimetria
  **conhecida e declarada**, não escondida: está na §4 de
  `docs/comparacao_modelos.md`. Uniformizar exigiria rodar a busca para os três — o
  que é barato para árvore e KNN e proibitivo para a MLP (216 configurações × 5
  folds). E a assimetria não muda a conclusão: mesmo a árvore podada (0,2450) fica
  0,046 atrás do boosting, muito acima do limiar de 0,005.
- *(OLS/Ridge)* *"Por que manter os dois se dão o mesmo número?"* → **É esse o
  resultado.** Se OLS ≡ Ridge até a quinta casa e nenhum `alpha` de 0,1 a 1.000 muda
  nada, está provado que o gargalo do linear não é variância de coeficiente, é forma
  funcional. E descarta a hipótese de que a colinearidade do one-hot estivesse
  inflando os coeficientes.

### 6.6 Perguntaram do CRITÉRIO DE DECISÃO e você fez um modelo

- *"Como escolheram o vencedor?"* → Critério escrito **antes**, implementado em
  `decisao.py`: menor MAE de CV, e a vantagem só é **declarada** se as cinco folds
  concordarem **e** a diferença média for ≥ 0,005. Deu 5/5 e 0,0563 — vantagem
  declarada.
- *"Como sabemos que o critério não foi ajustado depois de ver o resultado?"* → O
  script **não abre nenhuma coluna `*_teste`**, por construção, e o resultado é
  gravado em `decisao_criterio.json` antes de qualquer número de teste ser lido. Há
  testes automatizados para os três desfechos: vantagem declarada, empate por fold
  discordante e empate por margem abaixo do limiar.
- *"E se tivesse dado empate?"* → Desempate declarado na ordem **explicabilidade →
  custo de previsão → número de hiperparâmetros**, com as duas primeiras ordens
  tabeladas no código (árvore é a mais explicável; OLS/Ridge/árvore são as mais
  baratas por previsão; KNN é a mais cara).
- *"Por que 0,005 e não outro número?"* → Porque o desvio entre folds das melhores
  configurações do projeto fica entre 0,0032 e 0,0043. Um limiar abaixo de um desvio
  declararia vantagem em ruído — e isso já aconteceu conosco: a busca de
  hiperparâmetros chegou a eleger uma configuração por 0,0005 num eixo que não tinha
  efeito nenhum.

### 6.7 As perguntas "pegadinha" (para qualquer um)

- *"Vocês usaram o conjunto de teste quantas vezes?"* → Uma, para relatar. A seleção
  de modelo e a busca de hiperparâmetros usaram só a CV do treino. Temos o caso
  documentado em que, depois de ver o teste, a configuração escolhida era pior em
  reais — e **não voltamos atrás**, porque isso seria usar o teste para selecionar.
- *"Por que MAE e não RMSE/R²?"* → Reportamos os três, mas decidimos por MAE em log.
  RMSE em reais é dominado pela cauda (um imóvel de R$ 19,8 mi), e a mesma assimetria
  que motivou o log motiva o MAE. Também reportamos o **erro percentual mediano**,
  que é a métrica que um corretor entenderia.
- *"O R² de 0,897 é em log ou em reais?"* → Em **log**, e dizemos isso sempre. Em
  reais seria outro número e menos honesto, porque o `exp()` amplifica erro na cauda.
- *"Melhorou o modelo ou melhorou o dado?"* → Nos dois casos em que a base mudou,
  medimos por **A/B nas mesmas linhas e mesmas folds**. Nos repasses, dos 0,0059 de
  "melhora" relatada só **0,0031** era o modelo prevendo melhor; o resto era o
  conjunto de avaliação ter perdido linhas impossíveis por construção. A afirmação
  correta é "a base ficou certa".
- *"Alguma hipótese de vocês estava errada?"* → **Cinco**, e são o melhor do
  trabalho: (1) corrigir o vazamento estrutural ia custar acurácia — melhorou; (2) o
  erro ia se concentrar no alto padrão — as duas pontas erravam; (3) `venda_direta`
  ia render 0,005–0,015 — rendeu 0,0010; (4) purificar Cabo Branco ia melhorar —
  piorou, e piorou porque ficou honesto; (5) KNN e MLP iam ficar entre Ridge e
  boosting — ficaram atrás dos dois. Registrar a hipótese antes é **regra executável
  no código**: o `dataclass` recusa candidato sem hipótese.
- *"E se os dados forem de uma janela de tempo específica?"* → É a nossa limitação
  mais séria e está declarada: nenhum JSON bruto tem campo de data, então **não há
  separação temporal**. Todo o resultado é interpolação dentro do mesmo instante de
  coleta.
- *"Se eu rodar de novo, dá o mesmo número?"* → **Na mesma máquina, sim; em outra,
  quase.** Medimos: o Ridge, que é solução fechada, devolve folds diferentes em
  ambientes diferentes, porque o `requirements.txt` declara `scikit-learn>=1.3.0` sem
  fixar versão e o particionamento do `GroupKFold` não é estável entre versões. O
  deslocamento máximo é **0,0013**, contra um limiar de 0,005 e uma distância de
  0,056 entre 1º e 2º — **o ranking e a decisão são idênticos**. Está documentado na
  §10 de `docs/comparacao_modelos.md`, com a correção definitiva (fixar as versões e
  rerodar a cadeia uma vez) registrada como issue própria. A semente garante o split;
  ela não garante a versão da biblioteca.
- *"O boosting ganhou por ser melhor algoritmo mesmo?"* → **Em parte.** Boosting e
  árvore reconstroem sozinhos a interação `bairro × area_util`; Ridge, OLS, KNN e
  MLP não têm como, e essa coluna **ainda não está explícita** na matriz. Já medimos
  esse efeito uma vez, com `venda_direta`: tornar explícita uma feature que só o
  não-linear inferia deu 4,5× mais ganho ao Ridge que ao boosting. Então parte da
  distância é engenharia de atributos, não capacidade — e registramos isso como
  limitação conhecida, não como resolvido.

---

## 7. Limitações — o que admitir sem hesitar

Admitir estas seis custa menos que ser pego em qualquer uma delas.

1. **Sem separação temporal.** Nenhum JSON bruto tem data. Validade externa
   limitada.
2. **`iptu` é proxy do alvo** — é calculado sobre o valor venal, ou seja, é função do
   preço. Não é vazamento temporal (está no anúncio), mas infla a performance de um
   jeito que não se sustenta para imóvel novo sem IPTU lançado. Presente em só 20%
   da base.
3. **A extração via LLM não cobre o zap.** Por isso `origem_anuncio` fica no modelo
   **como variável de controle** — sem ela, o modelo aprenderia a diferença entre
   portais achando que é diferença entre imóveis.
4. **O early stopping do boosting usa validação interna não agrupada.** O
   `HistGradientBoostingRegressor` liga `early_stopping='auto'` sozinho acima de
   10.000 amostras (temos 12.214) e separa 10% **aleatoriamente**. É a única parte do
   pipeline que o agrupamento não cobre; o efeito é limitado — decide só o momento de
   parada, não seleção de atributos. (Na MLP, que é do mesmo tipo, desligamos.)
5. **Não usamos validação cruzada aninhada.** Custaria 5× o tempo para responder uma
   pergunta que o teste já responde, já que ele foi separado antes da busca.
6. **A interação `bairro × area_util` não é coluna explícita** — parte da vantagem
   das árvores sobre os lineares é engenharia de atributos implícita.
7. **Os candidatos não estão no mesmo estágio de ajuste.** Só `ridge` e
   `gradient_boosting_ajustado` foram inscritos com a configuração vencedora de
   busca; árvore, KNN e MLP entraram na configuração inicial dos donos. A tabela lê
   "cada modelo como foi inscrito", não "cada modelo no seu melhor".
8. **As versões das bibliotecas não estão fixadas** (`scikit-learn>=1.3.0`), e o
   particionamento do `GroupKFold` muda entre versões — duas pessoas rodando a mesma
   base obtêm números até 0,0013 diferentes. Abaixo do limiar de 0,005, sem efeito no
   ranking, mas é o furo que resta na regra "mesmo split, mesmas folds".

---

## 8. Checklist antes de apresentar

- [x] ~~**Alinhar o número da árvore.**~~ Resolvido por documentação: `arvore.md`
      abre com uma tabela **[A] sem poda (inscrita, 0,2846)** vs **[B] podada
      (vencedora da busca, 0,2450)**, dizendo qual entra na tabela oficial e por quê;
      `comparacao_modelos.md` §4 e §7 declaram a assimetria. **Quem apresentar a
      árvore precisa saber os dois números** — está em §6.5 deste roteiro.
- [x] ~~**`mlp.md` com números de execução anterior.**~~ Resolvido: a tabela do
      `mlp.md` foi atualizada para a rodada #25 e ganhou nota de procedência
      apontando `resultados_modelos.csv` como fonte oficial. `resultados_mlp.csv`
      **continua no repo** como artefato da branch `feat/MLP` — foi mantido de
      propósito, para não apagar o entregável de outra pessoa; se preferirem removê-lo,
      é uma linha.
- [x] ~~`knn.md` afirmava ter o maior desvio entre folds.~~ Corrigido: o maior é o da
      MLP (0,0081); o KNN é o segundo (0,0064).
- [ ] **O código do scraper do ZapImóveis não está na `main`.** A pasta
      `src/imoveis_jp/scraping/zap_imoveis/` contém só um JSON — nenhuma linha de
      código. Os 13 módulos (~1.760 linhas) vivem em `origin/feat/zapimoveis`, uma
      branch **órfã** (sem ancestral comum com a main). Metade da coleta do projeto
      está fora do repositório principal; se pedirem para mostrar o scraper, ele não
      abre. Ver o plano de migração ao fim desta seção.
- [ ] **Fixar as versões** em `requirements.txt` (`==` em vez de `>=`) e, numa issue
      própria, rerodar `train.py → decisao.py → pca_variant.py` uma única vez. Só
      depois disso os números do repo voltam a ser reproduzíveis entre máquinas.
      **Não faça isso na véspera da apresentação:** muda todos os números publicados
      no README, em `modelagem.md` e em `protocolo_comparacao.md`.
- [ ] Trocar `dev A` / `dev E` / `dev KNN` / `dev OLS` / `dev (feat/MLP)` pelos
      **nomes reais** no campo `dono` dos candidatos.
- [ ] O README ainda lista a etapa 5 como "próxima etapa" — marcar como concluída e
      apontar para `docs/comparacao_modelos.md`.
- [ ] **Versionar as três figuras da árvore.** `arvore_curva_overfitting.png`,
      `arvore_podada.png` e `arvore_importancia_nativa.png` estão referenciadas em
      `arvore.md` mas **não estão no git** — existem só na máquina de quem gerou.
      Para o resto do time e para quem clonar o repo, as três imagens estão
      quebradas. `git add docs/figuras/arvore_*.png` resolve.
- [ ] Conferir que as demais figuras abrem: `residuos_diagnostico.png`,
      `importancia_permutacao.png`, `tsne_precos.png`, `heatmap_top30.png`.
- [ ] Rodar `pytest` uma vez na frente de todo mundo, para ninguém ser pego por teste
      quebrado.
- [ ] Cada dono relê a **hipótese** do próprio candidato — a professora pode pedir
      para recitar e comparar com o medido.
- [ ] Todo mundo decora a **seção 1** e o **cartão de bolso da seção 5**.

### 8.1 Plano de migração do scraper do ZapImóveis

Levantamento feito sobre `origin/feat/zapimoveis`. **A boa notícia é que os módulos
já são package-ready:** todo import interno usa o padrão
`try: from .config import X / except: from config import X`, então funcionam
inalterados dentro de um pacote.

| Módulo | Linhas | | Módulo | Linhas |
|---|---|---|---|---|
| `scraper.py` | 399 | | `tracker.py` | 143 |
| `extractor.py` | 231 | | `config.py` | 104 |
| `storage.py` | 179 | | `verifier.py` | 91 |
| `logger.py` | 159 | | `__main__.py` | 88 |
| `collector.py` | 78 | | `rate_limiter.py` | 74 |
| `controller.py` | 69 | | `url_builder.py` | 59 |

**Total: 13 módulos, ~1.760 linhas.** Dependências (`bs4`, `playwright`) já estão no
`requirements.txt`. **Não** levar junto: os 12 `.pyc`, o `scraping.log`, o
`pause.flag`, a cópia antiga do scraper do chaves em `src/scrapping/` e o
`imoveis_joao_pessoa_zap.json` de 189 mil linhas — que já está na main, em dois
lugares.

**Custo A — só mover (≈15 min, risco baixo).** Copiar os 13 `.py` para
`src/imoveis_jp/scraping/zap_imoveis/`. Ajustes necessários: **três linhas** em
`tests_resilience.py`, que importa pelo caminho velho
(`from src.scrapping.zap_imoveis...`) — ou deixar esse arquivo de fora. Validação:
`python -c "from imoveis_jp.scraping.zap_imoveis import scraper"`. Resultado: o
scraper entra no repo e roda, mas continua gravando dentro de `src/`.

**Custo B — conformidade com a regra nº 1 do README (+30–60 min).** Hoje o
`config.py` do zap deriva tudo de `os.path.dirname(os.path.abspath(__file__))`:

```python
DIR_ATUAL         = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_SAIDA     = os.path.join(DIR_ATUAL, "imoveis_joao_pessoa_zap.json")
PAUSE_FLAG_FILE   = os.path.join(DIR_ATUAL, "pause.flag")
STORAGE_STATE_FILE= os.path.join(DIR_ATUAL, "session_state.json")
```

Ou seja, o scraper **grava dados dentro do código** — e é literalmente por isso que
existe hoje um `imoveis_joao_pessoa_zap.json` dentro de
`src/imoveis_jp/scraping/zap_imoveis/`. A regra do README é que todo I/O de dados
passe por `imoveis_jp.config`. São ~4 linhas no `config.py` do zap mais uma
constante nova (`ANUNCIOS_ZAP_JSON = RAW / "imoveis_joao_pessoa_zap.json"`) em
`src/imoveis_jp/config.py` — que hoje **não tem** nenhuma constante do zap.

**A ressalva honesta:** o custo B não é verificável sem rodar o scraper de verdade
(precisa de playwright, rede e o site no ar). Dá para garantir que os imports
resolvem e que os caminhos apontam para onde devem; não dá para garantir uma coleta
ponta a ponta antes da apresentação. Por isso o **A** é o que eu recomendo agora, com
o **B** registrado como issue — melhor um scraper no repo com um defeito conhecido e
documentado do que meio scraper fora dele.

---

## 9. Sugestão de divisão de fala (15 min)

| Tempo | Quem | O quê |
|---|---|---|
| 0–1 | qualquer um | Pitch da seção 1 + o diagrama de 11 passos |
| 1–3 | Vitor Reis + João Victor | Coleta: 2 portais, resumível/shardável, ética/robots |
| 3–5 | Gabriel | Enriquecimento: descoberta empírica, LLM vs regex, **a validação de 94%** e o que decidimos não extrair |
| 5–6 | Micael + J. V. Sampaio | Features: 231 → 76, os três geradores, as duas features removidas por vazamento |
| 6–8 | J. V. Sampaio | Split agrupado, pipeline sem vazamento, CV e o limiar de 0,005 |
| 8–10 | J. V. Sampaio | Diagnóstico: regressão à média, permutação, **e os três defeitos de dado que ele revelou** |
| 10–12 | todos, 20 s cada | Cada um apresenta o próprio modelo: hipótese registrada → medido → veredito |
| 12–13,5 | J. V. Sampaio | Critério de decisão em código, vantagem declarada, PCA e t-SNE |
| 13,5–15 | qualquer um | Limitações (seção 7) e conclusão |

**Fechamento sugerido:** *"O melhor modelo erra 15,6% na mediana, e a vantagem dele
foi declarada por um critério que escrevemos em código antes de ver qualquer
resultado — num script que nem abre a coluna do teste. Mas o que mais nos ensinou
não foi isso: foi descobrir que 14% dos bairros estavam errados, que os anúncios
mais baratos não eram imóveis baratos, e que cinco das nossas previsões estavam
erradas — uma delas por dez vezes. O protocolo que registra a hipótese antes de
rodar é o que transformou cada um desses erros em resultado."*
