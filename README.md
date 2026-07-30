# Projeto — Paradigmas de Aprendizagem de Máquina

Previsão/análise de preços de **apartamentos à venda em João Pessoa (PB)**, a partir
de anúncios coletados do chavesnamao.com.br.

---

## Estrutura

```
Projeto-Aprendizagem-Maquina/
├── data/
│   ├── raw/          snapshots brutos do scrape        (versionado)
│   ├── interim/      resultados intermediários         (fora do git)
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
| Rodar os testes | `-m pytest` |

Detalhes do scrape (retomada, sharding, flags, ética/robots.txt): [docs/scraping.md](docs/scraping.md).

---

## Próximas etapas

1. Limpeza e tratamento da base → `src/imoveis_jp/processing/`
2. Extração via LLM das características que só existem na descrição
3. One-hot + matriz de correlação para enxugar atributos → `src/imoveis_jp/features/`
4. Treino e avaliação dos modelos → `src/imoveis_jp/models/`

---

## Pendências conhecidas

- O módulo `enrich_from_description.py` e seu teste **perderam o código-fonte** na
  reorganização anterior; sobraram apenas os `.pyc`, preservados em `.recuperar/`
  (fora do git). Dá para descompilar ou reescrever do zero.
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
