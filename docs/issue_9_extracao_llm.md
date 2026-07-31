# Documentação Técnica e Acadêmica da Issue #9: Extração de Atributos e Diferenciais Exóticos via LLM (Groq API)

**Autor:** Gabriel Ribeiro (`@gabrielbribeiroo`)  
**Projeto:** Previsão e Análise de Preços de Imóveis em João Pessoa (PB)  
**Disciplina:** Paradigmas de Aprendizagem de Máquina — UFPB  
**Módulo:** `src/imoveis_jp/processing/extract_llm_features.py`  
**Data:** 30/07/2026  

---

## 1. Justificativa da Seleção de Atributos e Captura de Diferenciais Exóticos

Em vez de definir atributos arbitrários ou fixos, realizou-se uma análise amostral sobre as descrições brutas e comodidades do dataset (`data/raw/imoveis_joao_pessoa.json`).

Constatou-se que corretores e proprietários frequentemente omitiram dados nos campos estruturados do formulário HTML, inserindo informações cruciais para a precificação apenas no texto livre da descrição.

### 📋 Atributos Selecionados e Justificativa de Escolha:

1. **`distancia_praia_m` (Numérico / Metros):**
   - *Motivação:* Em João Pessoa, a proximidade com o mar (ex: "150m da praia", "300m do mar") é um dos fatores de maior impacto no preço por $m^2$. O formulário HTML do portal não possui campo numérico de distância em metros.
2. **`posicao_solar` (Categórico: Nascente / Poente / Sul / Norte):**
   - *Motivação:* A orientação solar determina o nível de ventilação e incidência de calor à tarde no litoral paraibano. Imóveis *Nascente* têm valorização superior.
3. **`status_construcao` (Categórico: Na planta / Em construção / Pronto p/ morar / Usado):**
   - *Motivação:* Identifica a fase do imóvel. Imóveis em construção ou na planta costumam ter preços abaixo do valor de mercado pronto.
4. **`tipo_unidade` (Categórico: Térreo com área / Térreo simples / Cobertura / Duplex / Tipo):**
   - *Motivação:* Apartamentos térreos com área privativa externa (área própria) possuem precificação diferente de apartamentos em andares intermediários.
5. **`vista_mar` / `beira_mar` (Booleanos):**
   - *Motivação:* Distingue imóveis que possuem apenas vista para o mar daqueles localizados na avenida beira-mar (pé na areia).
6. **`moveis_projetados` (Booleano):**
   - *Motivação:* Identifica imóveis com armários embutidos e móveis planejados instalados, agregando valor à venda.
7. **`reformado` (Booleano):**
   - *Motivação:* Identifica imóveis usados que passaram por atualização completa de acabamento.
8. **`aceita_permuta` / `aceita_fgts` (Booleanos):**
   - *Motivação:* Condições comerciais que ampliam o público comprador.
9. **`diferenciais_unicos` (Lista Dinâmica de Strings):**
   - *Motivação:* **Previne a perda de atributos exóticos ou raros.** Captura recursos como *"pé direito duplo"*, *"tomada para carro elétrico"*, *"automação residencial"*, *"jacuzzi"*, *"painéis solares"* ou *"solário"*.

---

## 2. Estratégia de Processamento em Lote (Batching)

Para permitir o processamento dos 10.758 imóveis no mesmo dia no plano gratuito do Groq Cloud sem perda de acurácia:
* **Tamanho do Lote:** 5 imóveis por chamada de API (`--batch-size 5`).
* **Redução de Chamadas:** Reduziu 10.758 chamadas para apenas **~2.150 requisições**.
* **Tempo Total Estimado:** ~35 a 45 minutos.

---

## 3. Resiliência do Pipeline

* **Tratamento de Rate Limits (HTTP 429):** Algoritmo de *Exponential Backoff* com Jitter.
* **Salvamento Atômico de Checkpoint (`extractions_llm.json`):** Salva o progresso a cada lote concluído de forma atômica e resumível.

---

## 4. Guia de Execução no Terminal

```powershell
# 1. Executar a extração dos atributos e diferenciais exóticos para a base inteira:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --batch-size 5

# 2. Executar um teste com amostra de 10 imóveis:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --limit 10 --batch-size 5

# 3. Exportar o resultado salvo em extractions_llm.json para CSV:
.\.venv\Scripts\python.exe -m imoveis_jp.processing.extract_llm_features --merge
```
