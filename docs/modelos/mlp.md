# Modelo MLP (Multi-Layer Perceptron) — Resumo Prático

---

## 1. O que o Algoritmo MLP Faz?

O **MLP (Multi-Layer Perceptron)** é uma **Rede Neural Artificial** composta por camadas de neurônios interconectados. 

* **Entrada**: Ele recebe as 76 características do imóvel (bairro, metragem, número de quartos, vagas, comodidades como piscina, elevador, etc.).
* **Processamento Interno**: Os dados passam por 2 camadas ocultas com neurônios (100 neurônios na 1ª camada e 50 na 2ª). Cada neurônio calcula combinações dos dados e aplica uma função de ativação (`ReLU`) para aprender padrões e interações complexas entre os atributos.
* **Saída**: O algoritmo calcula uma previsão contínua para o preço de venda do imóvel.

---

## 2. Resultados Empíricos Medidos

Abaixo estão os resultados do modelo **MLP** comparados aos outros algoritmos testados no projeto sobre dados inéditos de teste em João Pessoa:

| Modelo | Erro % Mediano | Erro Médio (R$) | Capacidade de Explicação ($R^2$) | Desempenho Relativo |
|---|---|---|---|---|
| **Gradient Boosting Ajustado** | **15,5%** | R$ 169.450 | **89,7%** | 🥇 Campeão |
| **Gradient Boosting (Padrão)** | **16,2%** | R$ 172.856 | **89,2%** | 🥈 2º Lugar |
| **Ridge (Linear)** | **19,1%** | R$ 249.200 | **84,4%** | 🥉 3º Lugar |
| **MLP (Rede Neural)** | **20,8%** | **R$ 286.046** | **74,6%** | 4º Lugar |
| **Baseline (Chute na Mediana)** | **43,1%** | R$ 443.793 | −0,2% | Pior piso |

### Conclusão dos Resultados:
* A **MLP erra em média 20,8%** no valor do imóvel (cerca de R$ 286 mil em média).
* Ela explica **74,6%** das variações de preço do mercado imobiliário.
* Supera com folga o chute cego (43,1% de erro), mas fica atrás do **Gradient Boosting** (15,5% de erro). Isso ocorre porque redes neurais sofrem para ajustar gradientes em tabelas contendo muitas variáveis binárias esparsas (comodidades), enquanto árvores de decisão particionam esse tipo de dado com muito mais facilidade.

---

## 3. Decisões Técnicas Práticas

1. **Escala Obrigatória (`escalar_binarias=True`)**:
   Todas as variáveis (contínuas e binárias) foram padronizadas na mesma escala. Sem isso, as variáveis contínuas dominariam o gradiente da rede e o aprendizado travaria.
2. **Convergência (`max_iter=500`)**:
   O limite de épocas foi subido para 500 para garantir que a rede tivesse tempo de treinar sem interromper o aprendizado pela metade.
3. **Validação Íntegra (`GroupKFold`)**:
   O treino foi avaliado em 5 fatias de dados agrupadas por imóvel físico, impedindo que imóveis duplicados aparecessem ao mesmo tempo no treino e na validação.
4. **Configuração Campeã da MLP**:
   * **Ativação**: `ReLU` (evita o travamento do aprendizado).
   * **Solver**: `Adam` (ajusta a velocidade de aprendizado por atributo).
   * **Arquitetura**: 2 camadas `(100, 50)`.

---

## 4. Resumo de Conformidade

* **Arquivo do modelo**: `src/imoveis_jp/models/candidatos/mlp.py`
* **Testes automatizados**: 105/105 aprovados (`pytest`).
* **Regras de convivência**: NENHUM arquivo compartilhado do projeto foi alterado.
