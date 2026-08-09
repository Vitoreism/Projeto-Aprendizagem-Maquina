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
| Gradient Boosting ajustado | 0,2155 | R$ 167.866 | 16,2% | 0,868 |
| Gradient Boosting (padrão) | 0,2238 | R$ 171.392 | 17,0% | 0,861 |
| Ridge | 0,2906 | R$ 290.742 | 22,3% | 0,766 |
| Baseline (mediana) | 0,6531 | R$ 417.354 | 42,5% | −0,001 |

Metodologia, decisões e limitações: [docs/modelagem.md](docs/modelagem.md).

### Etapa 4b — resíduos e importância por permutação

`analysis` mede, **no conjunto de teste**, onde o modelo erra e do que ele
depende. Três resultados:

- **O modelo puxa tudo para o meio.** O viés vai de +6,9% no quintil mais barato
  a −14,1% no mais caro, trocando de sinal monotonicamente. As duas pontas são as
  piores faixas (20,0% e 19,5% de erro mediano) contra 13,1% no centro.
- **`bairro` e `area_util` sozinhos valem 0,42 dos 0,61 de importância total.**
  24 dos 75 atributos têm importância indistinguível de zero.
- **Correlação não é importância.** `com_lavabo` é a 11ª maior correlação com o
  preço e vale zero para o modelo; `bairro_bessa` tem correlação 0,012 e é uma
  das dummies mais úteis. A primeira é efeito de área e bairro vazando por uma
  proxy; a segunda só funciona em interação — que é por que o boosting ganha do
  Ridge.

A análise também expôs um problema de dado: os 16 anúncios abaixo de R$ 50 mil
erram +132% na mediana porque não são preços de venda (R$ 603/m² contra uma
mediana de R$ 9.019/m²). Detalhes e o que fazer: §9 de
[docs/modelagem.md](docs/modelagem.md).

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
