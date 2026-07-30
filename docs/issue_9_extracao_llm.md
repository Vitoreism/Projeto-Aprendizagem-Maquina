# Documentação Técnica e Acadêmica da Issue #9: infraestrutura de Extração de Texto Livre via LLM (Groq API)

**Autor:** Gabriel Ribeiro (`@gabrielbribeiroo`)  
**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)  
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB  
**Módulo:** `src/imoveis_jp/processing/extract_llm_features.py`  
**Data:** 30/07/2026  

---

## 1. Introdução e Propósito do Módulo

A **Issue #9** tem como propósito criar a infraestrutura robusta, resiliente e extensível de **Processamento de Linguagem Natural (PLN) baseada em LLMs (Large Language Models)** para extrair atributos do texto livre do campo `descricao_completa` dos imóveis.

Nesta etapa, o foco principal é a **construção da arquitetura técnica**, **estimativa precisa de tokens/custos**, **controle de taxas de requisição (*Rate Limits*)** e **salvamento incremental em lote**, permitindo que a equipe posteriormente defina ou ajuste o schema de atributos desejado sem precisar reconstruir o pipeline.

---

## 2. Estimativa Científica de Tokens e Custos (Requisito da Issue #9)

Foi realizado um estudo estatístico detalhado sobre o corpus de descrições dos anúncios (`data/raw/imoveis_joao_pessoa.json`):

### 📊 2.1 Estatísticas do Dataset
* **Total de imóveis cadastrados:** `10.758` anúncios
* **Imóveis com descrição válida:** `10.751` anúncios (99.9%)
* **Média de caracteres por descrição:** `743.8` caracteres
* **Maior descrição:** `4.667` caracteres
* **Volume total de caracteres:** `7.996.962` caracteres (~8 milhões de caracteres)

### 🔢 2.2 Estimativa de Tokens
Usando a taxa de conversão para o português (~1 token = 3.7 caracteres):
* **Tokens das descrições brutas:** ~`2.161.341` tokens
* **Tokens de instruções do prompt (System + User):** ~`1.612.650` tokens (150 tokens/imóvel)
* **Tokens de saída estruturada (JSON):** ~`860.080` tokens (80 tokens/imóvel)
* **TOTAL ESTIMADO DE TOKENS DO PROJETO:** **~`4.634.071` tokens** (~4.63 milhões)

### ⚡ 2.3 Análise de Cotas e Custo (Groq Free Tier)
- **Modelo de referência:** `llama-3.1-8b-instant`
- **Custo financeiro:** **R$ 0,00** (Gratuito no Groq Cloud)
- **Cota Diária (TPD):** 500.000 tokens / dia
- **Cota de Requisições (RPD):** 14.400 requisições / dia (~30 RPM)
- **Tempo estimado de execução total:** ~7.5 dias no limite diário gratuito (ou ~2.5 dias se agrupado em batches).

---

## 3. Arquitetura do Pipeline Técnico

### 3.1 Tratamento de Rate Limits (HTTP 429) & Exponential Backoff
O pipeline trata flutuações e limites de requisições da API através do algoritmo de **Exponential Backoff com Jitter**:

$$\Delta t_{\text{espera}} = \min\left(t_{\text{máx}}, t_{\text{base}} \times 2^{\text{tentativa}}\right) + \text{uniform}(0.5, 1.5)$$

### 3.2 Persistência Incremental e Escrita Atômica
- **Arquivo de Checkpoint:** `data/interim/extractions_llm.json`
- **Resumível / Idempotente:** Pula automaticamente anúncios já processados em execuções anteriores.
- **Escrita Atômica:** Garante a integridade do JSON mesmo em caso de interrupção abrupta do processo.

---

## 4. Guia de Execução e Reprodutibilidade

Para rodar qualquer etapa no terminal na raiz do repositório:

```powershell
# 1. Estimativa automática de tokens e custos:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.estimate_llm_cost

# 2. Execução em modo seco (Dry-Run / sem chamada de API):
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --dry-run --limit 10

# 3. Teste pontual com limite de N imóveis:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --limit 5

# 4. Normalização para CSV intermediário:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --merge
```

---

## 5. Próximos Passos (Alinhamento com a Equipe)

Quando a equipe decidir no futuro quais atributos específicos deseja extrair da descrição livre para o modelo final, basta atualizar a constante `SYSTEM_PROMPT` e a validação do schema em `extract_llm_features.py` e reexecutar o pipeline.
