# Projeto — Paradigmas de Aprendizagem de Máquina

Previsão/análise de preços de **apartamentos à venda em João Pessoa (PB)**, a partir
de anúncios coletados do chavesnamao.com.br.

---

## Estrutura

```
Projeto-Aprendizagem-Maquina/
├── data/
│   ├── raw/          snapshots brutos do scrape        (versionado)
│   ├── interim/      resultados intermediários         (fora do git, exceto
│   │                 os checkpoints de extração via LLM: representam cota
│   │                 de API já consumida e não são regeneráveis de graça)
│   └── processed/    datasets finais em CSV            (versionado)
├── docs/             notas e decisões (docs/scraping.md)
├── notebooks/        exploração e EDA
├── scripts/          utilitários de linha de comando (.bat)
├── src/imoveis_jp/   o pacote Python do projeto
│   ├── config.py     caminhos canônicos — todo I/O de dados passa por aqui
│   ├── scraping/     coleta (chaves_na_mao: parser, scraper, merge_parts)
│   ├── processing/   limpeza e normalização raw → processed
│   ├── features/     engenharia de atributos / EDA
│   └── models/       treino e avaliação
├── tests/
├── pyproject.toml
└── requirements.txt
```

Duas regras que mantêm isso simples daqui em diante:

1. **Nenhum script usa caminho relativo ao diretório atual.** Tudo vem de
   `imoveis_jp.config`, então os comandos funcionam de qualquer pasta.
2. **Dado bruto nunca é editado no lugar.** O fluxo é sempre
   `data/raw` → `data/interim` → `data/processed`.

---

## Setup

No **PowerShell**, na raiz do repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m playwright install chromium
```

O `-e .` instala o pacote em modo editável: você edita `src/imoveis_jp/` e o
efeito é imediato, sem reinstalar e sem gambiarra de `sys.path`.

---

## Comandos

| O quê | Comando (com `.venv\Scripts\python.exe`) |
|---|---|
| Coletar anúncios | `-m imoveis_jp.scraping.chaves_na_mao.scraper` |
| Ver o plano sem baixar | `-m imoveis_jp.scraping.chaves_na_mao.scraper --dry-run` |
| Coletar em paralelo (3 workers) | `scripts\run_parallel.bat` |
| Fundir as partes do paralelo | `-m imoveis_jp.scraping.chaves_na_mao.merge_parts` |
| Normalizar para CSV | `-m imoveis_jp.processing.normalize_to_csv` |
| Extrair campos da descrição (diagnóstico) | `-m imoveis_jp.processing.enrich_from_description` |
| Consolidar one-hot e montar a matriz de features | `-m imoveis_jp.features.build_features` |
| Matriz de correlação e seleção de atributos | `-m imoveis_jp.features.correlation` |
| Treinar e avaliar os modelos | `-m imoveis_jp.models.train` |
| Buscar hiperparâmetros | `-m imoveis_jp.models.tune` |
| Resíduos e importância por permutação | `-m imoveis_jp.models.analysis` |
| Rodar os testes | `-m pytest` |

Detalhes do scrape (retomada, sharding, flags, ética/robots.txt): [docs/scraping.md](docs/scraping.md).

---

## Próximas etapas

1. ~~Limpeza e tratamento da base~~ → `src/imoveis_jp/processing/`
2. ~~Extração via LLM das características que só existem na descrição~~
3. ~~One-hot + matriz de correlação para enxugar atributos~~ → `src/imoveis_jp/features/`
4. ~~Treino e avaliação dos modelos~~ → `src/imoveis_jp/models/`
5. Comparar seis modelos sob protocolo pré-registrado → issues
   [#20](../../issues/20) (infra, bloqueante), [#21](../../issues/21) árvore,
   [#22](../../issues/22) KNN, [#23](../../issues/23) MLP,
   [#24](../../issues/24) OLS, [#25](../../issues/25) comparação final

### Etapa 3 — como funciona

O one-hot das comodidades vinha de três geradores independentes (LLM sobre a
descrição, HTML do chaves na mão, HTML do zap), o que produzia colunas duplicadas
e colinearidade embutida. `build_features` consolida tudo numa matriz só:

- reconstrói do JSON bruto as células numéricas de `suites` e `banheiros` que a
  fusão HTML+LLM tinha sobrescrito com `True`/`False`;
- preenche campos estruturados ausentes lendo a descrição (issue #3): `suites`
  deixou de faltar em 54,3% da base e passou a faltar em 46,9%
  ([docs/extracao_da_descricao.md](docs/extracao_da_descricao.md));
- unifica as binárias equivalentes por OR, tratando `NaN` como ausência — na base
  deduplicada, `NaN` significava "o outro portal não usa esse termo";
- descarta pseudo-atributos (`apartamento`, `lazer`, `conforto`…) e comodidades
  presentes em menos de 1% dos imóveis;
- anula valores fora da faixa plausível (havia preço de R$ 470 milhões);
- normaliza `posicao_solar`, `status_construcao`, `tipo_unidade`,
  `origem_anuncio` e `bairro` como **texto**. O one-hot delas acontece dentro do
  `Pipeline` de treino, não aqui — fazê-lo sobre a base inteira definia o
  conjunto de colunas usando as linhas que virariam teste.

`correlation` roda em cima da matriz e gera, em `data/processed/`, o ranking de
cada atributo contra o preço (`correlacao_alvo.csv`), a poda dos redundantes
(`pares_redundantes.csv`), a lista final (`features_selecionadas.csv`) e os
heatmaps em `docs/figuras/`.

Números, decisões e a pendência da extração via LLM no zap:
[docs/features_one_hot_correlacao.md](docs/features_one_hot_correlacao.md).

### Etapa 4 — como funciona

Split 80/20 **agrupado por imóvel físico** (o mesmo apartamento aparece em até 7
anúncios; com split aleatório ele cairia no treino e no teste), semente `42`,
alvo em `log(preco_venda)`. Imputação, padronização **e one-hot** ficam dentro de
um `Pipeline` do sklearn, reajustado em cada fold do `GroupKFold(5)` — nenhuma
estatística atravessa a fronteira treino/validação. O teste é tocado uma única vez.

| Modelo | CV MAE (log) | Teste MAE | Erro % mediano | R² (log) |
|---|---|---|---|---|
| Gradient Boosting ajustado | 0,1988 | R$ 170.150 | 15,6% | 0,897 |
| Gradient Boosting (padrão) | 0,2076 | R$ 172.892 | 16,1% | 0,892 |
| Ridge | 0,2551 | R$ 249.200 | 19,1% | 0,844 |
| Baseline (mediana) | 0,6183 | R$ 443.793 | 43,1% | −0,002 |

Metodologia, decisões e limitações: [docs/modelagem.md](docs/modelagem.md).

### Etapa 4b — resíduos e importância por permutação

`analysis` mede, **no conjunto de teste**, onde o modelo erra e do que ele
depende. Três resultados:

- **O modelo puxa tudo para o meio.** O viés vai de +7,9% no quintil mais barato
  a −14,3% no mais caro, trocando de sinal monotonicamente. O topo é a pior faixa
  (19,1% de erro mediano) contra 13,6% no melhor quintil.
- **`bairro` e `area_util` sozinhos valem 0,42 dos 0,61 de importância total.**
  34 dos 76 atributos têm importância indistinguível de zero.
- **Correlação não é importância.** `com_lavabo` é a 11ª maior correlação com o
  preço e vale zero para o modelo; `bairro_bessa` tem correlação 0,012 e é uma
  das dummies mais úteis. A primeira é efeito de área e bairro vazando por uma
  proxy; a segunda só funciona em interação — que é por que o boosting ganha do
  Ridge.

A análise também expôs dois problemas de dado, os dois corrigidos depois: o
bairro (etapa 4c) e o preço que não era preço (etapa 4d). Detalhes: §9 de
[docs/modelagem.md](docs/modelagem.md).

### Etapa 4c — canonização dos bairros

**14,0% dos anúncios tinham o bairro errado**, no atributo mais importante do
modelo. Três defeitos em `extrair_bairro`: casamento por substring na ordem
errada (511 anúncios do Altiplano Cabo Branco viravam Cabo Branco, bairros com
26% de diferença de preço/m²); um fallback que devolvia a primeira palavra do
endereço e inventava bairros como `avenida`, `doutor` e `telegrafista`; e nomes
não-canônicos.

A correção lê a estrutura do endereço — que difere entre os dois portais — e casa
contra os **64 bairros oficiais** de João Pessoa por conjunto de tokens, com o
nome mais específico vencendo. Nunca inventa: o que não casa vira
`nao_informado`, e sobraram 13 anúncios (0,08%) — 8 sem endereço e 5 fora do
município.

Sete localidades que não constam da lista oficial (Jardim Luna, Novo Milênio,
Colinas do Sul…) entraram numa lista à parte, confirmadas por conhecimento local
e sustentadas pelos dados: Jardim Luna tem CV de preço/m² de 0,23, mais
homogêneo que o bairro oficial mediano (0,35).

**329 valores distintos → 66.** Como `bairro` entra na chave de deduplicação, a
correção também revelou 579 duplicatas entre portais que antes escapavam: a base
cai de 16.162 para 15.583 linhas, sem perda de imóvel.

Medido em A/B sobre as mesmas linhas e folds: CV **0,2144 → 0,2097**, com os 5
folds concordando. Fechar a lista nos 64 oficiais e reajustar os
hiperparâmetros sobre a base nova levaram a CV a **0,2057**. E a matriz depois do one-hot
cai de 349 para 131 colunas, com o modelo melhor.

### Etapa 4d — o preço que não era preço

Os anúncios mais baratos da base não eram imóveis baratos: eram **repasses de
financiamento**. O valor anunciado é o ágio pago pelas chaves, e o comprador
ainda assume as parcelas — dois produtos com o mesmo rótulo `preco_venda`.
*"Repasse no Valentina: Chaves R$ 21.500 e Parcela Menor que Aluguel (R$ 719)"*.

A distribuição de preço/m² não tem vale, então o piso foi calibrado contra um
sinal independente: a palavra "repasse"/"ágio" no texto, que 177 anúncios
declaram. Em R$ 1.000/m², 76% dos descartados se autodeclaram. Em R$ 1.500 a
precisão cai para 50% — porque ali entra uma população **legítima**: 311 vendas
diretas/leilão da Caixa, com preço/m² entre R$ 1.164 e R$ 1.899. O piso fica
abaixo delas de propósito, ao custo declarado de 21 falsos positivos.

Nove anúncios tinham o defeito inverso — a **área** errada, o preço certo (988 m²
num anúncio intitulado "98m²"). Aí anula-se a área, não a linha, e a regra roda
antes do piso para que o preço válido não saia junto.

**A base cai de 15.583 para 15.476 linhas e a CV vai de 0,2057 a 0,1998** — mas o
A/B sobre as mesmas 3.087 linhas de teste mostra que o efeito real é de apenas
**+0,0031**, abaixo do limiar de 0,005 do projeto. O resto da "melhora" é o
conjunto de avaliação ter perdido linhas impossíveis por construção. A afirmação
correta não é que o modelo melhorou: é que a base ficou certa. §9.10 de
[docs/modelagem.md](docs/modelagem.md).

### Etapa 4e — a binária `venda_direta`

262 anúncios de venda direta/leilão de banco têm preço/m² mediano de R$ 1.665
contra R$ 9.089 do resto — **18% do mercado**. Os termos foram calibrados um a
um: `caixa economica` e `aceita fgts` foram reprovados (são opção de
financiamento em anúncio comum, preço/m² de mercado), e `matricula` foi
reprovado por redundância — os 5 anúncios que ele adiciona sozinho custam de
R$ 2.790 a R$ 8.694/m².

**Previsão registrada antes de rodar: ganho de 0,005 a 0,015. Medido: 0,0010.**
O erro tem explicação — o Ridge ganha 0,0045, 4,5 vezes mais que o boosting,
porque a árvore já reconstruía "apartamento pequeno *naquele* bairro" a partir
de `bairro` × `area_util`. Previ o ganho do modelo que não tinha como já saber.

A coluna é a **6ª mais importante** por permutação e mesmo assim removê-la custa
0,0010: permutação mede dependência, ablação mede insubstituibilidade, e a
diferença é a redundância. Importância alta não justifica manter uma feature.

E o ganho está onde não se esperava: no segmento o viés cai de +11,4% para
+4,1%, mas **77% do ganho global vem dos outros 3.040 anúncios** — sem a coluna,
os leilões puxavam para baixo a previsão de todo apartamento parecido.

Fica como aviso para a Etapa 5: **toda feature que só o modelo não-linear
consegue inferir sozinho enviesa a comparação a favor dele.** §9.11 de
[docs/modelagem.md](docs/modelagem.md).

### Etapa 5 — infraestrutura da comparação de modelos

`src/imoveis_jp/models/candidatos/` é um **registro por descoberta**: cada dev
cria um arquivo e exporta uma constante `CANDIDATO`. Não existe lista central
para editar, e é por isso que cinco pessoas trabalham na mesma semana sem
colidir — quando duas mexeram em `build_features.py` na mesma janela, quatro dos
cinco conflitos foram em arquivo gerado.

O `Candidato` **recusa** hipótese vazia. Não é burocracia: três dos achados mais
úteis do projeto foram previsões erradas (o vazamento estrutural que melhorou o
modelo, o erro que não estava só no alto padrão, o `venda_direta` que rendeu dez
vezes menos que o previsto). Nenhuma apareceria se a hipótese pudesse ser
escrita depois do resultado — então a regra virou código.

`ridge` e `gradient_boosting_ajustado` migraram para o formato novo, com os
números idênticos aos de antes (0,2551 e 0,1988), provando que a refatoração não
mexeu em nada.

Protocolo, as três decisões de projeto e o aviso sobre viés de comparação:
[docs/protocolo_comparacao.md](docs/protocolo_comparacao.md).

Por que a extração via LLM do zap continua pendente — e por que completá-la
provavelmente não vale a pena:
[docs/extracao_zap_diagnostico.md](docs/extracao_zap_diagnostico.md).

---

## Pendências conhecidas

- `data/raw/imoveis_joao_pessoa.json` (~15 MiB) **é versionado**, para que a etapa de
  limpeza parta exatamente da mesma base. Evite recommitá-lo sem necessidade: cada
  nova versão do arquivo adiciona uma cópia inteira ao histórico do repositório.
  Se um dia ele passar de ~50 MB, aí sim vale migrar para Git LFS.

---

## Uso dos dados

O `Content-Signal` do chavesnamao declara **`ai-train=no`**: o corpus **não** deve ser
usado para treinar ou fazer fine-tuning de modelos de linguagem. `search=yes` e
`ai-input=yes` (grounding/RAG) são permitidos, assim como a análise estatística e a
modelagem tabular deste trabalho acadêmico.
