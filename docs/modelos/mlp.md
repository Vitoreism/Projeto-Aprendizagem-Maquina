# Documentação do Candidato MLP (Multi-Layer Perceptron)

---

## 1. Motivação e Enquadramento Teórico

A arquitetura **Multi-Layer Perceptron (MLP / Rede Neural Feed-Forward)** faz parte do conteúdo didático de Aprendizagem de Máquina (Prova 2), onde é explicitamente caracterizada como um modelo não-linear capaz de realizar **classificação e regressão**.

Diferente de modelos lineares (como o OLS e o Ridge), que assumem hiperplanos de decisão aditivos, a MLP combina combinações lineares ponderadas através de funções de ativação não-lineares em múltiplas camadas ocultas:

$$\hat{y} = g^{(2)}\left( \mathbf{W}^{(2)} \cdot g^{(1)}\left( \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \right) + \mathbf{b}^{(2)} \right)$$

Esta capacidade permite que a rede aprenda aproximações universais de funções contínuas e capture interações complexas entre atributos (por exemplo, o impacto não-linear combinado de $\text{área\_útil} \times \text{bairro}$).

---

## 2. Expectativa vs. Realidade em Dados Tabulares

### Expectativa Teórica (Hipótese Prévia)
> *"A MLP captura interações não-lineares entre área e localização sem necessitar de termos cruzados manuais, devendo superar o Ridge (CV ~0,25). Contudo, em dados tabulares (~13k linhas), o otimizador por gradiente sofre com esparsidade e fronteiras de decisão contínuas, devendo ficar atrás do Gradient Boosting (CV ~0,20). Espera-se CV MAE(log) entre 0,21 e 0,23."*

### Resultados Empíricos Medidos

| Modelo | CV MAE (log) | CV MAE Desvio | Teste MAE (R$) | Erro % Mediano | R² (log) |
|---|---|---|---|---|---|
| **Gradient Boosting Ajustado** | **0,2002** | 0,0028 | R$ 169.450 | 15,5% | 0,897 |
| **Gradient Boosting (Padrão)** | **0,2087** | 0,0044 | R$ 172.856 | 16,2% | 0,892 |
| **Ridge** | **0,2553** | 0,0045 | R$ 249.200 | 19,1% | 0,844 |
| **MLP (Multi-Layer Perceptron)** | **0,3116** | 0,0091 | R$ 286.046 | 20,8% | 0,746 |
| **Baseline (Mediana)** | **0,6183** | 0,0097 | R$ 443.793 | 43,1% | −0,002 |

### Honestidade Intelectual & Fundamentação
Em problemas com dados tabulares de médio porte (~12.214 amostras de treino):
1. **Árvores de Decisão e Ensembles (Boosting)** particionam o espaço de entrada com cortes ortogonais diretos nos eixos dos atributos, o que se ajusta com alta eficiência à natureza discreta e esparsa de dados imobiliários (ex: presença/ausência de 60+ comodidades e dummies de bairros).
2. **Redes Neurais Densa (MLP)** tentam suavizar superfícies de decisão através de hiperplanos contínuos. A otimização por gradiente estocástico sobre uma matriz contendo 63 variáveis binárias esparsas apresenta elevado ruído e lentidão na convergência, obtendo CV MAE(log) de **0,3116** (Erro Mediano de 20,8%).

Portanto, os dados confirmam que a MLP supera expressivamente a baseline nula (CV MAE 0,6183), mas sofre em matrizes esparsas tabulares comparada a modelos baseados em árvores e mesmo ao Ridge regularizado.

---

## 3. Armadilhas Técnicas Conhecidas & Soluções Aplicadas

### Armadilha 1: Escala Total das Features (`escalar_binarias=True`)
* **Problema**: O pré-processamento padrão padroniza variáveis contínuas ($\mu=0, \sigma^2=1.0$), mas deixa variáveis binárias em $\{0, 1\}$. Uma variável binária com proporção $p=0.1$ tem variância $\sigma^2 = p(1-p) = 0.09$. Redes neurais utilizam retropropagação de erro baseada no gradiente das entradas:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \mathbf{\delta}^{(1)} \cdot \mathbf{x}^T$$
  Features não-escaladas com ordens de grandeza discrepantes fazem com que variáveis contínuas dominem a atualização dos pesos, desestabilizando o aprendizado.
* **Solução**: O candidato declara obrigatoriamente `escalar_binarias=True` no contrato do `Candidato`, garantindo que todas as 63 colunas binárias passem pelo `StandardScaler()` antes de alimentarem o `MLPRegressor`.

### Armadilha 2: Não-Convergência (`max_iter=200`)
* **Problema**: O valor padrão de `max_iter=200` no `scikit-learn` é insuficiente para otimizar os milhares de pesos da rede sobre a matriz de 76+ atributos, disparando o aviso `ConvergenceWarning`.
* **Solução**: O modelo base foi configurado com `max_iter=500` (ou superior na busca), permitindo que o solver alcance a tolerância de parada por gradiente nulo sem interrupções prematuras.

### Armadilha 3: Defeito Metodológico do `early_stopping=True`
* **Problema**: Ativar `early_stopping=True` faz o `MLPRegressor` separar internamente 10% dos dados de treino para validação através de uma amostragem **aleatória simples**.
* **Falha**: No mercado imobiliário local, um mesmo apartamento físico é frequentemente anunciado até 7 vezes por imobiliárias diferentes (com pequenas variações no texto). O split principal do projeto utiliza `GroupShuffleSplit` agrupado por imóvel físico para evitar que cópias do mesmo imóvel caiam simultaneamente em treino e teste. Se o `early_stopping` for ativado, linhas idênticas do mesmo imóvel são sorteadas para o conjunto interno de validação da rede, camuflando o sobreajuste e violando a regra de integridade §6 de `docs/modelagem.md`.
* **Decisão**: O `early_stopping=True` **não é utilizado** na avaliação final, mantendo o treinamento sobre todos os dados do fold e avaliando a generalização estritamente via `GroupKFold(5)`.

---

## 4. Análise Experimental dos Entregáveis

### A. Funções de Ativação (`ReLU` vs `tanh` vs `logistic`)
* **`relu` ($f(x) = \max(0, x)$)**: Apresenta a melhor taxa de convergência e previne o problema de gradiente desaparecente (*vanishing gradient*) em camadas mais profundas.
* **`tanh` ($f(x) = \tanh(x)$)**: Simétrica em torno de zero, mas sofre com saturação em valores extremos ($|x| > 2$), desacelerando o treino.
* **`logistic` ($f(x) = \frac{1}{1 + e^{-x}}$)**: Apresenta o pior desempenho na regressão de preços, devido ao esmaecimento acentuado dos gradientes durante a retropropagação em dados escalados.

### B. Solvers de Otimização (`adam` vs `sgd`)
* **`adam` (Adaptive Moment Estimation)**: Utiliza estimativas de primeiro e segundo momentos dos gradientes ($\beta_1, \beta_2$). Demonstra convergência muito mais rápida e estável em matrizes com termos esparsos (como comodidades binárias).
* **`sgd` (Stochastic Gradient Descent)**: Apresenta convergência extremamente lenta sem um ajuste fino complexo da taxa de aprendizado e momento, frequentemente estagnando em mínimos locais subótimos.

### C. Efeito da Taxa de Aprendizagem (`learning_rate_init`)
Conforme ressaltado na teoria:
> *"Se a função de perda começar a aumentar, diminua a taxa de aprendizado."*

* **Taxa ideal ($\eta \approx 0.001$)**: A função de perda diminui monotonicamente a cada época até a convergência suave.
* **Taxa excessiva ($\eta \ge 0.1$)**: O passo de atualização nos pesos excede o mínimo local na superfície de erro ($\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla \mathcal{L}$). A perda passa a oscilar violentamente ou aumentar a cada época, degradando o MAE final.

### D. Arquitetura de Camadas Ocultas (`hidden_layer_sizes`)
* **Arquiteturas rasas `(50,)` e `(100,)`**: Apresentam menor capacidade de representação, agindo de forma semelhante a modelos lineares com transformações simples.
* **Arquitetura profunda `(100, 50)`**: Permite a combinação hierárquica de atributos (camada 1 detecta padrões espaciais/bairros; camada 2 combina área útil e padrão do condomínio), obtendo o menor erro da família MLP.

---

## 5. Resumo da Conformidade com o Protocolo

| Critério / Regra | Conformidade | Observação |
|---|---|---|
| Arquivo em `candidatos/mlp.py` | ✅ SIM | Módulo isolado criado em `src/imoveis_jp/models/candidatos/mlp.py` |
| Hipótese prévia escrita | ✅ SIM | Declarada no contrato antes da execução da primeira CV |
| Integridade dos scripts do core | ✅ SIM | Nenhuma alteração em `train.py`, `dataset.py`, `build_features.py` ou `analysis.py` |
| Sem commit de arquivos gerados | ✅ SIM | `data/processed/` e `docs/figuras/` mantidos fora do git |
| Suíte de testes (Pytest) | ✅ PASSOU | 105/105 testes aprovados (`test_candidatos.py`) |
