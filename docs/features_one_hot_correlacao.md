# Etapa 3 — Consolidação do One-Hot e Matriz de Correlação

**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB
**Módulos:** `src/imoveis_jp/features/build_features.py` e `src/imoveis_jp/features/correlation.py`
**Data:** 01/08/2026

---

## 1. O problema de partida

O one-hot das comodidades **já existia** quando esta etapa começou — só que gerado
três vezes, por caminhos independentes que nunca conversaram entre si:

| Origem | Colunas | Gerador |
|---|---|---|
| LLM sobre a descrição | `piscina`, `academia`, `aceita_fgts`… (45) | `extract_llm_features.py` |
| HTML do chaves na mão | `comodidade_*`, com sinônimos mapeados (~103) | `extract_amenities_from_scrap.py` |
| HTML do zap | `comodidade_*`, **sem** normalização (~57) | `extract_llm_features.py` |

O resultado eram 231 colunas em `imoveis_joao_pessoa_global_deduplicated.csv`, das
quais 205 binárias, com seis defeitos que inviabilizavam a correlação:

1. **`NaN` não significava "não tem", significava "o outro portal".**
   `comodidade_piscina` era nulo em exatamente os 7.167 anúncios do zap;
   `comodidade_banheira` nos 6.473 do chaves. Por isso 144 colunas estavam como
   `object` em vez de `bool`.
2. **Colunas duplicadas.** O ramo do zap não passava por `normalizar_texto` nem
   por `mapear_sinonimos`, então conviviam `comodidade_deposito`/`comodidade_depósito`,
   `comodidade_piso_porcelanato`/`comodidade_porcelanato`,
   `comodidade_gas_canalizado`/`comodidade_gás_encanado`.
3. **Colinearidade perfeita embutida.** A fusão fazia `piscina = piscina_LLM OR
   comodidade_piscina`, então as duas colunas eram literalmente a mesma
   informação — não existia um único caso de `comodidade_piscina=True` com
   `piscina=False`.
4. **`suites` e `banheiros` corrompidos.** A mesma fusão casava `comodidade_<x>`
   com qualquer chave de mesmo nome, e `comodidade_suites`/`comodidade_banheiros`
   sobrescreviam as colunas **numéricas** com `True`/`False`: 7.006 células de
   `suites` e 8.988 de `banheiros`.
5. **Numéricos como texto:** `preco_venda` = `'625.000'`, `garagens` com `'--'`,
   `condominio`/`iptu` com `'Isento'` e `'Não informado'`.
6. **Esparsidade:** 105 colunas binárias apareciam em menos de 1% dos imóveis, e
   havia pseudo-atributos que não descrevem nada (`apartamento`, `sala`, `lazer`,
   `conforto`, `praticidade`).

---

## 2. Correções na origem

Em `extract_llm_features.py`, a fusão HTML+LLM passou a exigir que o nome seja de
fato um atributo booleano vindo da LLM antes de fundir. Sem esse filtro, qualquer
comodidade cujo nome coincidisse com o de uma coluna do scrap a destruía — foi o
que aconteceu com `suites` e `banheiros`.

O caminho de fallback do JSON do zap também foi corrigido: era uma string
relativa com escapes inválidos (`"...\src\imoveis_jp\scraping\..."`, que dispara
`SyntaxWarning: invalid escape sequence '\s'` no Python 3.13) e só resolvia se o
comando fosse executado de fora do repositório. Agora deriva de `config.ROOT`,
como manda a regra nº 1 do README.

As duas correções valem para as **próximas** execuções do pipeline. A base já
gerada é reparada na etapa seguinte.

---

## 3. `build_features.py` — a consolidação

Roda em cima de `imoveis_joao_pessoa_global_deduplicated.csv` e produz
`data/processed/features_matrix.csv`. **231 → 77 colunas**, sobre 15.476 imóveis.

### 3.1 Reparo das numéricas destruídas

Cada célula não-numérica é reconstruída a partir do JSON bruto, casando por
`url_anuncio` (a cobertura dos dois JSONs é de 100% das 15.476 linhas):

| Coluna | Células inválidas | Recuperadas |
|---|---|---|
| `suites` | 7.006 | 6.911 |
| `banheiros` | 8.988 | 8.826 |
| `garagens` | 649 | 0 (eram `'--'` na origem) |

### 3.2 Unificação das binárias

Cada coluna é reduzida a um nome canônico por `normalizar_texto` + `mapear_sinonimos`
e as equivalentes são fundidas por OR lógico, com `NaN` tratado como ausência.
**205 binárias → 157 canônicas, em 30 grupos fundidos.** Isso resolve de uma vez
os defeitos 1, 2 e 3 da seção 1.

Duas ressalvas importantes:

- **Prefixo obrigatório.** Toda binária sai como `com_<nome>`. Sem isso,
  `comodidade_suites` canonizaria para `suites` e colidiria de novo com a coluna
  numérica — exatamente o bug 4, reintroduzido pela própria correção.
- **Falsos positivos do casamento por substring.** `mapear_sinonimos` casa por
  `in`, então `carpete` caía em "pet" (→ `permitido_pets`) e `supermercados` em
  "mar" (→ `vista_ou_acesso_praia`). Ambos estão na lista de exceções.

### 3.3 Limpeza e filtros

- **Conversão numérica** com os conversores já existentes em
  `deduplicate_dataset.py`; `'Isento'` vira `0`, não nulo.
- **Limites de plausibilidade** — sem isso o Pearson era ruído puro: um único
  anúncio de 58 milhões de m² anulava a estatística. Anulados: `area_util` 134,
  `area_total` 70, `condominio` 35, `garagens` 10, `preco_venda` 8, `iptu` 5,
  `quartos` 2, `suites` 2, `banheiros` 1.
- **Frequência mínima de 1%** e remoção dos pseudo-atributos: 13 + 82 colunas
  descartadas, **restam 62 comodidades**.

### 3.4 O one-hot das categóricas não mora mais aqui

| Coluna | Tratamento |
|---|---|
| `posicao_solar`, `status_construcao`, `tipo_unidade`, `origem_anuncio` | normalizadas como **texto** |
| `bairro` | canonizado do endereço via `extrair_bairro` → **65 lugares curados** + `nao_informado` |
| `anunciante` (442 níveis) | **descartado** — ver §3.5 |

A versão anterior desta etapa fazia `pd.get_dummies` aqui, com dois cortes de
frequência: bairro com menos de 30 imóveis virava `outros` (sobravam 38 níveis) e
dummy abaixo de 1% era descartada. Os dois contavam linhas da **base inteira**,
incluindo as que virariam teste — **vazamento estrutural**.

O one-hot passou para dentro do `Pipeline` de treino (`OneHotEncoder` com
`min_frequency=30` e `handle_unknown='infrequent_if_exist'`), onde a contagem
acontece dentro de cada fold. Não custou acurácia; ganhou. Os números estão em
[§3.1 de modelagem.md](modelagem.md).

Os 329 níveis de bairro que existiam aqui eram, em boa parte, artefato de
extração: 9,8% dos anúncios estavam em categorias inventadas por um fallback que
pegava a primeira palavra do endereço. Depois da canonização contra a lista dos
64 bairros oficiais de João Pessoa (mais 7 localidades reconhecidas), restam
**65 com anúncio** e 0,08% sem bairro.
Ver [§9.7 de modelagem.md](modelagem.md).

Nesta matriz, portanto, **5 categóricas continuam 5 colunas de texto**. O corte
espelhado de dummy quase-constante (frequência > 99%) não foi reimplantado, e o
motivo está medido no comentário de `build_features.py`: as três colunas em
questão somam 0,04% do erro do modelo.

### 3.5 Duas features removidas por vazamento

Uma auditoria de pré-processamento derrubou duas colunas que existiam na primeira
versão desta matriz. O critério: **só entra aqui transformação determinística ou
com constante fixa**; qualquer coisa que aprenda estatística dos dados precisa
acontecer depois do split.

**`bairro_preco_m2_medio`** vinha de `neighborhoods.csv` e foi tratada como fonte
externa. Não é. Comparando o valor do arquivo com a mediana de preço/m² calculada
desta própria base, bairro a bairro:

| bairro | arquivo | nossa base |
|---|---|---|
| cabo_branco | 14.676 | 13.830 |
| altiplano_cabo_branco | 10.909 | 10.706 |
| bessa | 9.333 | 9.427 |
| estados | 7.472 | 7.533 |

**Correlação 0,996, erro relativo mediano 2,4%.** É uma agregação do alvo, e era a
6ª feature mais forte do ranking (Spearman +0,504). O efeito de bairro segue
coberto pela própria coluna `bairro`, que não toca no alvo — e a importância por
permutação mostra que ela é o atributo mais forte do modelo (+0,217 de MAE), ou
seja, nada de essencial se perdeu com a remoção.

**`anunciante_qtd_anuncios`** era `value_counts()` sobre a base inteira: a
contagem de cada anunciante incluía os anúncios que cairiam no teste.

O que **não** foi removido, e por quê: os limites de plausibilidade da §3.3 são
constantes de domínio escritas à mão, não estatísticas dos dados. Aplicá-los antes
do split é saneamento, não aprendizado. Se um dia virarem IQR ou z-score, terão de
migrar para dentro do `Pipeline`.

Pelo mesmo critério, `features_selecionadas.csv` **não deve ser usado como filtro
do treino**: a seleção usa a correlação com o alvo de todas as linhas. É artefato
de EDA e de relatório.

---

## 4. `correlation.py` — a matriz e a seleção

Alvo principal `preco_venda`, mais `log_preco` e `preco_m2` como alvos auxiliares
(preço imobiliário é fortemente assimétrico). Descarta os 175 imóveis sem preço;
restam 15.301 e 125 features candidatas.

Binária contra alvo contínuo é correlação ponto-bisserial e binária contra
binária é o coeficiente phi — as duas são a fórmula de Pearson, então uma matriz
só cobre os três casos.

### 4.1 Ranking (Spearman / Pearson contra `preco_venda`)

| Atributo | Spearman | Pearson |
|---|---|---|
| `area_util` | +0,696 | +0,737 |
| `suites` | +0,658 | +0,675 |
| `area_total` | +0,630 | +0,341 |
| `garagens` | +0,625 | +0,697 |
| `banheiros` | +0,533 | +0,594 |
| `quartos` | +0,503 | +0,491 |
| `condominio` | +0,319 | +0,235 |

### 4.2 Poda de redundância

Varre o triângulo superior e, em cada par com |r| ≥ 0,85, descarta o menos
correlacionado com o alvo. Só 1 par sobreviveu até aqui (`area_total` vs
`area_util`, r = 0,92), **99 → 98 features** — número baixo justamente porque a colinearidade pesada já tinha sido
eliminada na consolidação.

### 4.3 Diagnóstico de artefato de coleta

Os dois portais publicam vocabulários de comodidade diferentes, então uma
comodidade que só existe em um deles fica correlacionada com o **portal**, não com
o imóvel. O módulo sinaliza toda feature com |r| ≥ 0,50 contra `origem_anuncio_*`.

### 4.4 Saídas

| Arquivo | Conteúdo |
|---|---|
| `data/processed/features_matrix.csv` | a matriz consolidada, 15.476 × 77 |
| `data/processed/correlacao_alvo.csv` | ranking de cada feature contra os três alvos |
| `data/processed/pares_redundantes.csv` | decisões da poda, com o motivo |
| `data/processed/features_selecionadas.csv` | as 123 features que sobraram (uso de EDA — ver §3.5) |
| `data/interim/relatorio_consolidacao.json` | auditoria da consolidação |
| `docs/figuras/heatmap_top30.png` | top 30 por correlação com o preço |
| `docs/figuras/heatmap_completo.png` | matriz das features selecionadas |

---

## 5. Pendência conhecida: a extração via LLM não cobre o zap

O detector da seção 4.3 acusou `com_piscina` com r = −0,57 contra o zap. A causa
é maior do que um problema de vocabulário: **a extração via LLM nunca rodou sobre
a base do zapimoveis.**

Em `imoveis_joao_pessoa_zap_master.csv`, `posicao_solar` é "Nao informado" em
11.833 dos 11.841 anúncios, `tipo_unidade` é "Apartamento tipo" em 100% deles e
`diferenciais_unicos` está vazio em todos menos um. Os 8 registros com algum valor
real são sobra de um teste com `--limit`. O checkpoint `extractions_llm_zap.json`
não existe.

O efeito prático: `com_piscina` marca 48,1% dos anúncios do chaves e 0,6% dos do
zap. O ramo "descrições" do one-hot cobre só um dos dois portais.

Enquanto isso não for resolvido, **mantenha `origem_anuncio_*` como variável de
controle no modelo** — já está na matriz. Para resolver, é preciso um `.env` com
`GROQ_API_KEYS` e:

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --dataset zap
.\.venv\Scripts\python.exe -m imoveis_jp.processing.deduplicate_dataset
.\.venv\Scripts\python.exe -m imoveis_jp.features.build_features
.\.venv\Scripts\python.exe -m imoveis_jp.features.correlation
```

O ganho é limitado: **78% dos anúncios do zap não têm descrição utilizável**
(ausente ou com menos de 10 caracteres), então uma execução completa alcançaria
cerca de 2,6 mil anúncios — 16% da base global.

---

## 6. Execução

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.features.build_features
.\.venv\Scripts\python.exe -m imoveis_jp.features.correlation
```

Ambos os módulos são idempotentes e reconstroem tudo do zero a cada execução.
Os testes em `tests/test_features.py` travam os bugs desta etapa — a colisão de
nomes, o `NaN` que significava portal e os falsos positivos por substring.
