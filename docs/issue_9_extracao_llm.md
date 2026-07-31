# Documentação Técnica e Acadêmica da Issue #9: Processamento em Lote (Batching) via LLM (Groq API)

**Autor:** Gabriel Ribeiro (`@gabrielbribeiroo`)  
**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)  
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB  
**Módulo:** `src/imoveis_jp/processing/extract_llm_features.py`  
**Data:** 30/07/2026  

---

## 1. Introdução e Estratégia de Execução no Mesmo Dia

A **Issue #9** tem como propósito a extração de características em texto livre do campo `descricao_completa` dos anúncios através de Processamento de Linguagem Natural (PLN) guiado por LLMs.

Para viabilizar o processamento de todos os **10.758 imóveis no mesmo dia** dentro do plano gratuito do Groq (Free Tier), adotou-se a **Estratégia de Lote (Batching)**.

---

## 2. A Estratégia de Lote (Batching)

### 🚀 2.1 Como Funciona
Em vez de realizar uma chamada de API para cada anúncio individualmente (o que geraria 10.758 requisições e atingiria a cota diária em 10% do progresso), o script agrupa **5 a 10 descrições de imóveis no mesmo prompt de envio**.

A LLM recebe a lista de imóveis identificados por um `id_lote` e retorna estritamente uma matriz JSON de respostas:

```json
{
  "resultados": [
    { "id_lote": 0, "posicao_solar": "Nascente", "vista_mar": true, "beira_mar": false, ... },
    { "id_lote": 1, "posicao_solar": "Poente", "vista_mar": false, "beira_mar": false, ... }
  ]
}
```

### 📉 2.2 Ganhos de Desempenho e Eficiência
* **Redução de Requisições:** De 10.758 chamadas para apenas **~1.070 a 2.150 chamadas de API**.
* **Economia de Tokens:** Redução de **~90% no overhead de envio do System Prompt**.
* **Tempo Total de Execução:** Reduzido de ~24 horas para **apenas ~35 a 60 minutos no mesmo dia**!

---

## 3. Resiliência do Pipeline (Rate Limits e Checkpoints)

1. **Exponential Backoff (HTTP 429):** Se a API retornar um limite de requisição por minuto (Rate Limit), o script aguarda automaticamente com tempo exponencial antes de tentar novamente.
2. **Escrita Atômica de Checkpoints (`extractions_llm.json`):** A cada lote concluído, o resultado é salvo no disco de forma segura e atômica. Se o processo for interrompido, ao executar novamente ele **pula todos os imóveis já processados** e continua exatamente de onde parou.

---

## 4. Guia de Execução no Terminal

```powershell
# 1. Executar a extração em lote para toda a base no mesmo dia (batch-size=5):
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --batch-size 5

# 2. Executar um teste com amostra menor (ex: 20 imóveis):
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --limit 20 --batch-size 5

# 3. Exportar o resultado salvo do checkpoint para o CSV intermediário:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --merge
```
