# Coleta de dados — chavesnamao

Ferramenta de **curadoria de dados**: coleta anúncios reais de imóveis que servem de
base para a modelagem. Todos os comandos abaixo assumem que você está na **raiz do
repositório** e já rodou o setup do [README](../README.md).

> **robots.txt do chavesnamao (verificado):** páginas de anúncio são permitidas.
> O `Content-Signal` do site declara **`ai-train=no`** (⚠️ **não** usar o conteúdo
> para treinar/fine-tunar modelos de linguagem), `search=yes` e **`ai-input=yes`**.
> **Não reutilize o corpus para treinar LLMs.**

---

## 1. Rodar o scrape

Domínio do projeto: **apartamentos à venda em João Pessoa (~10.755 anúncios)**.

```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.scraper
```

Padrões: `apartamento` / `venda` / `pb-joao-pessoa`, salvando em
`data/raw/imoveis_joao_pessoa.json`, com sleeps educados de 3–7 s entre anúncios.

### Para deixar rodando a noite toda

- **Deixe a janela do PowerShell aberta** e **desative a suspensão** do Windows
  (Configurações → Sistema → Energia → "Tela e suspensão" → Suspensão = Nunca
  quando na tomada). Se o PC dormir, o scrape pausa.
- Quer salvar um log? Acrescente `| Tee-Object scrape.log` no fim do comando.

### ⏱️ Quanto demora

Com sleeps de 3–7 s + carga da página, dá **~8–10 s por anúncio** →
**~10,7k anúncios ≈ 24–30 h** (mais de uma noite). Não tem problema: é **resumível**.

---

## 2. Rodar em PARALELO (mais rápido) 🚀

Divide o trabalho em N workers, cada um numa janela do PowerShell. Cada worker pega
uma **fatia disjunta** e escreve no **seu próprio arquivo**
(`data/raw/imoveis_joao_pessoa.parteXdeN.json`), pulando o que já está no arquivo
canônico. No fim você **funde** tudo.

Recomendo **3 workers** (~0,5 req/s no total, tranquilo). Corta o tempo de ~14 h → **~5 h**.

**Confira a divisão primeiro (sem baixar nada):**
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.scraper --shard 0/3 --dry-run
```

**Jeito fácil — abre as 3 janelas de uma vez:**
```powershell
.\scripts\run_parallel.bat
```

**Ou manualmente**, em 3 janelas do PowerShell (só muda o número antes da `/3`):
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.scraper --shard 0/3
.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.scraper --shard 1/3
.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.scraper --shard 2/3
```

**Quando os 3 terminarem (ou você parar todos com Ctrl+C), funda:**
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.merge_parts
```
Isso junta as partes em `data/raw/imoveis_joao_pessoa.json` (sem duplicar). É idempotente.

> Cada worker é **resumível** igual ao modo simples: `Ctrl+C` salva, mesmo comando
> retoma. Quer 4 workers? Use `--shard 0/4 … 3/4`. Se notar páginas vindo vazias
> (bloqueio), reduza pra 2 workers ou aumente `--min-sleep/--max-sleep`.

---

## 3. Parar e retomar (o pulo do gato)

- **Parar:** `Ctrl+C` na janela. Ele salva o progresso antes de sair.
- **Retomar:** rode **exatamente o mesmo comando** de novo. Ele relê o JSON,
  **pula tudo que já foi salvo** e continua de onde parou. Pode fazer isso quantas
  noites forem necessárias.

O arquivo é salvo a cada 25 anúncios novos (escrita atômica — não corrompe se cair).

---

## 4. Acompanhar o progresso

- No console aparece um checkpoint a cada 25: `💾 checkpoint: N salvos | i/total | ETA h:mm:ss`.
- Para contar a qualquer momento (outra janela, na raiz do repo):

```powershell
.\.venv\Scripts\python.exe -c "import json; from imoveis_jp import config; print(len(json.loads(config.ANUNCIOS_JSON.read_text(encoding='utf-8'))), 'anuncios salvos')"
```

---

## 5. Ajustes úteis (opcionais)

| Flag | Para quê |
|---|---|
| `--max 500` | Testar / fatiar: pega só os 500 primeiros. |
| `--min-sleep 1.5 --max-sleep 3` | Mais rápido (~metade do tempo), porém mais risco de bloqueio. |
| `--min-sleep 6 --max-sleep 12` | Mais devagar e seguro. |
| `--out novo.json` | Começar um arquivo novo em vez de continuar o atual. |
| `--transacao aluguel` / `--tipo casa` | Mudar o escopo (fora do domínio do projeto). |
| `--refresh-links` | Recoletar a lista de links do sitemap (ignora o cache). |
| `--headful` | Mostrar o navegador (debug). |

Ver todas: `.\.venv\Scripts\python.exe -m imoveis_jp.scraping.chaves_na_mao.scraper --help`

---

## 6. Depois do scrape

1. **Limpeza/tratamento** da base → `src/imoveis_jp/processing/`.
2. **Extração via LLM** das características que faltam na descrição.
   ⚠️ O `enrich_from_description.py` que fazia isso teve o fonte perdido; sobrou
   só o `.pyc` em `.recuperar/`.
3. **One-hot + matriz de correlação** para enxugar atributos → `src/imoveis_jp/features/`.
4. **Normalização** para os CSVs:
   `.\.venv\Scripts\python.exe -m imoveis_jp.processing.normalize_to_csv`
   (lê `data/raw/`, escreve `data/processed/`).

> Se o bloco de anúncios do site mudar de layout, os campos podem vir vazios —
> nesse caso é ajustar os seletores em
> [src/imoveis_jp/scraping/chaves_na_mao/parser.py](../src/imoveis_jp/scraping/chaves_na_mao/parser.py).
