# Documentação da Issue #9: Extração de Características de Texto Livre via LLM

**Autor:** Gabriel Ribeiro (`@gabrielbribeiroo`)  
**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)  
**Disciplina:** Paradigmas de Aprendizagem de Máquina  
**Data:** 30/07/2026  

---

## 1. Contexto e Motivação Acadêmica

No processo de coleta de dados de portais imobiliários (como o *Chaves na Mão*), a extração de dados estruturados via HTML/BeautifulSoup recupera apenas os campos padrão declarados nas tags (como quantidade de quartos, banheiros, vagas de garagem e área útil). 

No entanto, ricas informações qualitativas e diferenciais de mercado estão presentes exclusivamente em formato não estruturado dentro do texto livre do campo **`descricao_completa`** de cada anúncio. Exemplos dessas características incluem:
- **Posição solar:** Nascente / Sul / Poente;
- **Vista:** Vista para o mar / beira-mar;
- **Acabamento:** Piso em porcelanato, móveis projetados / armários embutidos;
- **Comodidades exclusivas:** Varanda gourmet, área de lazer privativa, andar alto/baixo;
- **Condições:** Imóvel reformado, apto para financiamento, aceita permuta.

Para que esses atributos possam ser incorporados como *features* numéricas/categóricas nos modelos de Aprendizagem de Máquina do projeto (One-Hot Encoding / Vetorização), faz-se necessária uma etapa de **Processamento de Linguagem Natural (PLN) baseada em LLMs (Large Language Models)**.

---

## 2. Estimativa de Tokens, Custos e Limites de API (Requisito da Issue #9)

Conforme instruído na especificação da Issue #9, realizou-se um estudo prévio do volume de dados para dimensionar os requisitos computacionais, custos financeiros e limites de taxa de requisições (*Rate Limits*).

### 📊 2.1 Estatísticas do Dataset (`data/raw/imoveis_joao_pessoa.json`)
* **Total de imóveis coletados:** `10.758` anúncios
* **Imóveis com descrição não vazia:** `10.751` anúncios (99.9%)
* **Tamanho médio por descrição:** `743.8` caracteres
* **Maior descrição:** `4.667` caracteres
* **Volume total de texto:** `7.996.962` caracteres (~8 milhões de caracteres)

### 🔢 2.2 Estimativa de Tokens
Considerando a taxa de conversão média para a língua portuguesa (~1 token a cada 3.7 caracteres):
* **Tokens das descrições brutas:** ~`2.161.341` tokens
* **Overhead de prompt (instruções do sistema + exemplos):** ~`1.612.650` tokens (150 tokens/imóvel)
* **Tokens de saída (respostas estruturadas em JSON):** ~`860.080` tokens (80 tokens/imóvel)
* **VOLUME TOTAL ESTIMADO DE TOKENS:** **~`4.634.071` tokens** (~4.63 milhões)

---

## 3. Análise de Viabilidade e Estratégia de Execução

### 🌐 Opção A: API do Groq Cloud (Free Tier) — Modelo `llama-3.1-8b-instant`
- **Custo financeiro:** **R$ 0,00** (Plano Gratuito).
- **Limites da API Gratuita (*Free Tier*):**
  - **TPD (Tokens Por Dia):** 500.000 tokens/dia.
  - **RPM / RPD (Requisições):** 30 requisições por minuto / 14.400 requisições por dia.
- **Análise do tempo de execução:**
  - Enviando **1 imóvel por requisição**, o volume diário atinge o teto em ~1.100 a 1.200 imóveis/dia, necessitando de **~7.5 dias** para concluir a base inteira sem custos.
  - Agrupando **3 imóveis por requisição (Batching)**, o tempo diminui para **~2.5 a 3 dias**.

### 💻 Opção B: Execução Local (Ollama)
- **Modelos indicados:** `llama3.2:3b` ou `qwen2.5:3b`.
- **Vantagem:** Sem limite de cota diária por API ou tempo de espera de rate-limit.
- **Desvantagem:** Depende do poder de processamento (GPU/CPU) da máquina local.

---

## 4. Reprodutibilidade (Como Executar o Código)

O código de cálculo estatístico e estimativa foi encapsulado no módulo Python `imoveis_jp.processing.estimate_llm_cost`.

Para rodar a verificação a qualquer momento:

```powershell
# Execução no ambiente virtual do projeto (.venv)
.\.venv\Scripts\python.exe -m imoveis_jp.processing.estimate_llm_cost
```

---

## 5. Próximos Passos da Implementação (Issue #9)

1. **Desenvolvimento do Prompt em Formato JSON Schema:**
   Garantir que a LLM retorne estritamente um JSON padronizado com as variáveis booleanas/categóricas desejadas (ex: `{"nascente": true, "varanda_gourmet": false, "vista_mar": true}`).
2. **Construção do Pipeline de Lote (*Batching & Retry*):**
   Criar o script de extração com controle automático de requisições, tratamento de erro HTTP 429 (Rate Limit Exceeded) e persistência incremental dos resultados em `data/interim/extractions_llm.json`.
3. **Integração no Dataset Final:**
   Normalizar as variáveis extraídas para os CSVs finais em `data/processed/`.
