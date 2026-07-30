# Documentação Técnica e Acadêmica da Issue #9: Extração de Características em Texto Livre via LLM (Groq API)

**Autor:** Gabriel Ribeiro (`@gabrielbribeiroo`)  
**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)  
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB  
**Módulo:** `src/imoveis_jp/processing/extract_llm_features.py`  
**Data:** 30/07/2026  

---

## 1. Introdução e Formulação do Problema

Na mineração de dados imobiliários a partir de portais da web, a extração tradicional baseada em raspagem HTML (DOM Parsing) recupera com facilidade atributos tabulares padronizados (como número de quartos, banheiros, vagas de garagem e área útil). 

Contudo, uma parcela significativa do valor do imóvel é determinada por **atributos qualitativos que aparecem exclusivamente no texto livre da descrição completa** fornecida pelo anunciante. Exemplos clássicos no mercado imobiliário de João Pessoa incluem:
- **Orientação solar:** Posição Nascente (mais valorizada por evitar o calor da tarde no litoral), Poente, Sul ou Norte.
- **Localização em relação ao mar:** Beira-mar / Pé na areia, ou Vista definitiva para o mar.
- **Nível de acabamento:** Piso em porcelanato, móveis projetados / armários planejados.
- **Tipo de pavimento:** Andar alto vs. andar baixo / Cobertura.
- **Condições comerciais:** Aceita permuta, apto para financiamento bancário.

Para transformar esses dados não estruturados em variáveis numéricas e booleanas utilizáveis em modelos de Aprendizagem de Máquina (Regressão, Random Forest, XGBoost), foi desenvolvido o pipeline de **Processamento de Linguagem Natural (PLN) guiado por LLMs (Large Language Models)**.

---

## 2. Requisito 1: Especificação do JSON Schema e Prompt do Sistema

Para garantir que a LLM retorne estritamente respostas estruturadas e previsíveis (sem alucinações ou marcações de texto livre), o pipeline utiliza o parâmetro `response_format={"type": "json_object"}` e um **System Prompt com Schema rígido**.

### 📋 Tabela de Atributos Extraídos

| Chave JSON | Tipo | Valores Válidos | Descrição / Regra de Extração |
| :--- | :--- | :--- | :--- |
| `posicao_solar` | `string` | `"Nascente"`, `"Poente"`, `"Sul"`, `"Norte"`, `"Nao informado"` | Posição do sol. Deve ser extraída apenas se citada explicitamente. |
| `vista_mar` | `boolean` | `true` / `false` | `true` se mencionar vista para o mar, vista mar ou vista definitiva. |
| `beira_mar` | `boolean` | `true` / `false` | `true` se for localizado na avenida beira-mar ou pé na areia. |
| `varanda_gourmet` | `boolean` | `true` / `false` | `true` se possuir varanda/sacada/terraço gourmet. |
| `piso_porcelanato` | `boolean` | `true` / `false` | `true` se o acabamento incluir piso em porcelanato. |
| `moveis_projetados` | `boolean` | `true` / `false` | `true` se incluir armários embutidos, móveis planejados ou projetados. |
| `andar_alto` | `boolean` | `true` / `false` | `true` se mencionar andar alto, cobertura ou últimos pavimentos. |
| `reformado` | `boolean` | `true` / `false` | `true` se for imóvel reformado, atualizado ou recém-construído/novo. |
| `aceita_permuta` | `boolean` | `true` / `false` | `true` se mencionar aceite de troca, permuta ou veículos no negócio. |
| `aceita_financiamento` | `boolean` | `true` / `false` | `true` se a documentação estiver apta para financiamento bancário. |
| `ar_condicionado` | `boolean` | `true` / `false` | `true` se possuir ar condicionado, split ou infraestrutura para tal. |
| `area_lazer_privativa` | `boolean` | `true` / `false` | `true` se possuir piscina privativa, churrasqueira privativa ou área de lazer do apartamento. |

---

## 3. Requisito 2: Arquitetura do Pipeline em Lote (Rate Limits, Backoff & Checkpoints)

### 3.1 Tratamento de Erros e Exponential Backoff (HTTP 429)
APIs de LLM em nuvem possuem limites rígidos de requisições por minuto (**RPM**) e tokens por minuto (**TPM**). Quando a cota temporária é atingida, a API retorna um erro `HTTP 429 Too Many Requests`.

Para tratar esses cenários de forma resiliente e não interromper a execução, o módulo implementa o algoritmo de **Exponential Backoff com Jitter (Ruído Aleatório)**:

$$\Delta t_{\text{espera}} = \min\left(t_{\text{máx}}, t_{\text{base}} \times 2^{\text{tentativa}}\right) + \text{uniform}(0.5, 1.5)$$

Onde:
- $t_{\text{base}} = 2.0\text{s}$
- $\text{tentativa} \in \{0, 1, 2, 3, 4\}$
- O ruído aleatório ($\text{jitter}$) evita o efeito *thundering herd* se houver concorrência.

### 3.2 Persistência Incremental e Salvamento Atômico de Checkpoints
O processamento da base inteira (~10.758 imóveis) pode levar várias sessões devido à cota do plano gratuito do Groq.

- **Arquivo de Checkpoint:** `data/interim/extractions_llm.json`
- **Idempotência:** Antes de chamar a API para um imóvel, o pipeline verifica se a `url_anuncio` já está cadastrada no dicionário de checkpoints. Se estiver, o item é pulado instantaneamente.
- **Escrita Atômica:** A cada 10 novos itens processados, os dados são gravados primeiro em um arquivo temporário `extractions_llm.tmp` e renomeados de forma atômica (`replace()`), prevenindo a corrupção do JSON em caso de queda de energia ou interrupção pelo usuário (`Ctrl + C`).

---

## 4. Requisito 3: Integração e Normalização para o Dataset Final

Os resultados intermediários armazenados em `data/interim/extractions_llm.json` são convertidos em um DataFrame tabular limpo através da função `fundir_extracoes_nos_csvs_processados()`.

### Fluxo de Dados:
```text
data/raw/imoveis_joao_pessoa.json (Snapshots brutos)
       │
       ▼
[Groq LLM Pipeline: extract_llm_features.py]
       │
       ▼
data/interim/extractions_llm.json (Checkpoint JSON)
       │
       ▼
data/interim/llm_features_normalized.csv (Dataset tabular normalizado)
       │
       ▼
data/processed/properties.csv (Merge final para a modelagem ML)
```

---

## 5. Guia de Execução (Reprodutibilidade)

Todos os comandos devem ser executados no terminal na raiz do repositório utilizando o ambiente virtual `.venv`:

### 1. Teste de Validação em Modo Seco (sem gastar cota de API):
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --dry-run --limit 10
```

### 2. Teste de Extração com Amostra Reduzida (ex: 5 imóveis):
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --limit 5
```

### 3. Execução Completa (Resumível):
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features
```

### 4. Normalização dos Resultados para CSV:
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --merge
```

### 5. Reexecução do Cálculo de Estimativa de Tokens/Custos:
```powershell
.\.venv\Scripts\python.exe -m imoveis_jp.processing.estimate_llm_cost
```
