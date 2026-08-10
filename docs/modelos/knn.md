> Documentação escrita durante a consolidação da Fase 2 (issue #25) porque não
> havia sido entregue com o candidato (issue #22, fechada sem o markdown).
> Números tirados de `data/processed/resultados_modelos.csv` e
> `cv_mae_por_fold.csv`, gerados por esta mesma issue — nenhum valor inventado.

## KNN (K-Nearest Neighbors)

**Dono:** dev KNN · **Arquivo:** `src/imoveis_jp/models/candidatos/knn.py`

### O que este modelo assume sobre os dados

Regressão por vizinhos mais próximos: assume que o preço é **localmente
suave** no espaço de atributos — imóveis com área, bairro e comodidades
parecidos têm preços parecidos. Não tem forma funcional explícita; a previsão
é a média (ponderada por distância) dos vizinhos mais próximos no treino.

### Hipótese registrada ANTES de rodar

> "Deve ganhar do OLS/Ridge se o preço for localmente suave no espaço de
> features (vizinhos com área, bairro e comodidades parecidos têm preços
> parecidos), capturando interações sem modelá-las. Espera-se CV MAE(log)
> entre Ridge (~0,25) e o boosting (~0,20): melhor que linear, ainda atrás do
> boosting porque a alta dimensionalidade do one-hot dilui a distância
> euclidiana. Se ficar perto ou pior que Ridge, a maldição da
> dimensionalidade está matando a noção de vizinhança."

### Grade testada, e por que essa faixa

```
regressor__n_neighbors: [5, 11, 21, 41]
regressor__weights: ["uniform", "distance"]
```

`n_neighbors` cobre o eixo principal do modelo: com ~12 mil amostras de
treino e one-hot expandido, k pequeno demais memoriza ruído local, k grande
demais aproxima a média global e perde o sinal de bairro/comodidades.
`weights=distance` é o outro eixo útil — pesa vizinhos mais próximos mais
sem mudar a geometria do k escolhido. `escalar_binarias=True`, justificado
no próprio candidato: sem padronizar as binárias, as contínuas (variância ~1
após o `StandardScaler`) dominam a distância euclidiana sobre comodidades
0/1 de variância tipicamente <0,25.

### Resultado

| métrica | valor |
|---|---|
| CV MAE(log), média ± desvio | 0,3202 ± 0,0064 |
| Teste MAE (R$) | R$ 269.182 |
| Teste erro % mediano | 23,8% |
| Teste R² (log) | 0,715 |

Folds individuais (MAE log): 0,3256 · 0,3289 · 0,3192 · 0,3160 · 0,3113 — o
maior desvio entre folds de todos os seis candidatos (0,0064), sinal de que
o modelo é o mais sensível a qual conjunto de imóveis cai em cada fold.

### Por que esse resultado

A hipótese errou na direção: KNN não ficou entre Ridge e o boosting — ficou
**atrás de Ridge** (0,3202 vs 0,2551) e foi o **pior dos seis candidatos**,
inclusive atrás da árvore sem ensembling (0,2846). O próprio texto da
hipótese já previa esse desfecho como diagnóstico: "se ficar perto ou pior
que Ridge, a maldição da dimensionalidade está matando a noção de
vizinhança" — e foi isso que aconteceu. Com 132 colunas pós-one-hot (a
maioria binária e esparsa — dezenas de dummies de bairro, das quais só uma é
1 por imóvel), a distância euclidiana entre dois imóveis de bairros
diferentes fica dominada por um punhado de coordenadas 0-vs-1 que carregam
pouca informação de preço isoladamente, e "vizinho mais próximo" deixa de
significar "imóvel parecido" para significar, em boa parte, "imóvel do mesmo
bairro, independente do resto".

### A hipótese se confirmou?

Não na direção principal (não ficou entre Ridge e o boosting), mas confirmou
exatamente o diagnóstico de contingência que a própria hipótese registrou
para esse cenário — a maldição da dimensionalidade. É o segundo tipo de
achado que o protocolo do projeto valoriza (`docs/protocolo_comparacao.md`):
uma prisão certa sobre o mecanismo, mesmo quando a direção do resultado
principal erra.

### Limitação conhecida

KNN é o único dos seis candidatos cujo custo de previsão cresce com o
tamanho da base (precisa varrer, ou indexar, o treino inteiro a cada
previsão nova) — desvantagem estrutural que nenhuma configuração de `k` ou
`weights` resolve, e que pesa contra ele mesmo nos casos de empate técnico
(ver `EXPLICABILIDADE`/`CUSTO_PREVISAO` em `src/imoveis_jp/models/decisao.py`).
