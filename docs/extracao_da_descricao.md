# Issue #3 — Extração de campos estruturados a partir da descrição

**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)
**Módulo:** `src/imoveis_jp/processing/enrich_from_description.py`
**Data:** 02/08/2026

---

## 1. O problema

Muitos anunciantes não preenchem os campos estruturados do portal e jogam tudo na
descrição em texto livre. O scraper lê só o HTML estruturado, então esses imóveis
chegam com `quartos`, `suites`, `banheiros` ou `garagens` nulos — justamente as
variáveis mais preditivas do modelo:

| Campo | Ausência antes | Spearman com o preço |
|---|---|---|
| `suites` | 54,3% | +0,658 (2ª mais forte) |
| `garagens` | 5,4% | +0,625 |
| `banheiros` | 1,7% | +0,533 |
| `quartos` | 1,1% | +0,503 |

Dos 16.162 anúncios da base de então, **10.656 (65,9%) têm descrição utilizável**
— mais de 40 caracteres. (A canonização dos bairros depois levou a base a 15.583
linhas, ao revelar duplicatas entre portais; a proporção não muda de forma
relevante.)

---

## 2. Regra, não LLM

A issue admite "LLM ou qualquer outra técnica de NLP". A extração aqui é por
expressão regular, por três motivos práticos: é determinística (o mesmo texto dá
sempre o mesmo resultado, ao contrário de uma chamada a modelo com temperatura),
é testável em CI sem chave de API, e roda em segundos em vez de horas.

Os padrões aceitam dígito e número por extenso — "dois quartos", "três suítes"
aparecem cerca de 900 vezes na base. O texto é normalizado antes (minúsculas,
sem acento, espaços colapsados) e o valor extraído passa pelos mesmos limites de
plausibilidade do resto do pipeline, para que um número de telefone não vire 200
quartos.

**Não há risco de vazamento:** cada linha é lida isoladamente, sem nenhuma
estatística agregada, então a extração pode rodar antes do split.

---

## 3. Validação — o que autoriza usar isto

Antes de preencher qualquer campo ausente, a precisão foi medida **onde o campo
estruturado já existe**. Sem essa medida, o preenchimento seria um chute com
aparência de dado.

| Campo | Comparáveis | Acerto exato | Até ±1 |
|---|---|---|---|
| `garagens` | 5.903 | 94,4% | 99,0% |
| `suites` | 5.234 | 93,6% | 98,5% |
| `quartos` | 8.008 | 90,4% | 98,9% |
| `banheiros` | 1.332 | 86,0% | 95,6% |

### Por que `area_util` ficou de fora

Nenhum padrão de área passou de **67% de acerto**, nem com contexto explícito:

| Padrão | n | Dentro de 10% |
|---|---|---|
| `(\d+)\s*m²` simples | 4.719 | 65,0% |
| `área privativa/útil ... N m²` | 577 | 43,8% |
| `apartamento/imóvel ... N m²` | 2.067 | 67,2% |

A diferença mediana é **zero**, então não é viés sistemático que dê para
corrigir com um offset — é dispersão. As descrições citam várias áreas
(privativa, total, lazer, terreno, outra unidade do prédio) e o primeiro número
que casa nem sempre é o do imóvel.

`area_util` é a feature mais forte do modelo (Spearman +0,696). Preencher 6,5%
de ausência injetando ~35% de erro nela sai pior do que manter o nulo, que o
`SimpleImputer` do pipeline já trata. **Decisão: não extrair.**

---

## 4. Resultado

Preenche **apenas células ausentes** — valor informado pelo portal sempre manda.
Roda depois dos limites de plausibilidade, de propósito: célula anulada por
implausibilidade também pode ser recuperada do texto.

| Campo | Recuperadas | Ausência antes | Ausência depois |
|---|---|---|---|
| `suites` | 1.186 | 54,3% | **46,9%** |
| `garagens` | 112 | 5,4% | 4,7% |
| `quartos` | 37 | 1,1% | 0,8% |
| `banheiros` | 8 | 1,7% | 1,6% |

### Efeito no modelo: pequeno

| | CV MAE (log) | Teste MAE |
|---|---|---|
| Sem enriquecimento | 0,2334 ± 0,0039 | R$ 179.489 |
| Com enriquecimento | **0,2306 ± 0,0034** | R$ 173.672 |

Cerca de **1,2% de melhora** no MAE da validação cruzada. Vale registrar que as
métricas de teste **não são diretamente comparáveis** entre as duas linhas: o
enriquecimento alterou `quartos`, `banheiros` e `garagens`, que entram na
assinatura física usada para agrupar o split, então o conjunto de teste mudou
(3.210 → 3.167 anúncios). O número honesto de comparação é o da CV sobre o
treino.

O ganho ser pequeno é esperado: o gradient boosting já lidava razoavelmente com
a ausência, e a mediana imputada não estava longe. O valor principal aqui é de
**completude do dado** — `suites` deixou de faltar em mais da metade da base —
e isso importa para a análise descritiva tanto quanto para o modelo.

---

## 5. Execução

Integrado ao pipeline, roda junto com a matriz de features:

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.features.build_features
```

Como diagnóstico isolado, com o relatório de precisão da §3:

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.enrich_from_description
```
