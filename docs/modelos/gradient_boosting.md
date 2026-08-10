> Documentação escrita durante a consolidação da Fase 2 (issue #25), pelo dono
> do candidato (`dono: "dev E (integrador)"` em `candidatos/gradient_boosting.py`)
> — não havia sido entregue como markdown próprio; o histórico de tuning já
> vivia em `docs/modelagem.md` §8 e nos comentários do próprio arquivo.

## Gradient Boosting (`HistGradientBoostingRegressor`)

**Dono:** dev E (integrador) · **Arquivo:** `src/imoveis_jp/models/candidatos/gradient_boosting.py`

Duas versões aparecem nas tabelas de resultado: **`gradient_boosting`**
(configuração padrão do scikit-learn, fixada em `train.py` como referência de
quanto a busca de hiperparâmetros rendeu) e **`gradient_boosting_ajustado`**
(o candidato desta ficha, com a configuração vencedora de quatro passadas de
`GridSearchCV`).

### O que este modelo assume sobre os dados

Ensemble de centenas de árvores rasas, cada uma corrigindo o resíduo da
anterior. Diferente do linear, não assume forma aditiva — cortes por limiar
em sequência capturam interação entre atributos (ex.: área × bairro) sem
precisar que ninguém escreva o termo cruzado.

### Hipótese registrada ANTES de rodar

> "É o favorito, e a comparação existe justamente para não aceitar isso sem
> prova. A vantagem esperada sobre os lineares vem de interação (área x
> bairro) e de não-linearidade na cauda de preço. Onde ele deve perder:
> explicabilidade contra a árvore única, e extrapolação acima de R$ 2
> milhões, onde há pouco dado. Espera-se CV próxima de 0,20."

### Grade testada, e por que essa faixa

```
regressor__learning_rate:      [0.05, 0.1]
regressor__max_iter:           [500]
regressor__max_leaf_nodes:     [31, 63, 127, 255]
regressor__min_samples_leaf:   [2, 3, 5, 10, 20, 50]
regressor__l2_regularization:  [0.0]
```

Cada eixo tem histórico próprio documentado nos comentários do arquivo (e em
`docs/modelagem.md` §8): `max_leaf_nodes` foi o único que mexeu de verdade no
resultado e precisou ser estendido até 255 depois de cair na borda da grade
inicial; `min_samples_leaf=5` venceu por estar dentro de 0,0009 do mínimo
medido (3), mas foi preferido por ser o ponto mais conservador dentro do
limiar de 0,005 do projeto; `l2_regularization` é eixo morto (0,2234 vs
0,2235 entre l2=0 e l2=1) e ficou fixado para não gastar configurações nele.

### Resultado

| modelo | CV MAE(log) | Teste MAE (R$) | Teste erro % | Teste R² (log) |
|---|---|---|---|---|
| `gradient_boosting_ajustado` | 0,1988 ± 0,0030 | R$ 170.150 | 15,6% | 0,897 |
| `gradient_boosting` (padrão) | 0,2076 ± 0,0056 | R$ 172.892 | 16,1% | 0,892 |

Folds do ajustado (MAE log): 0,1992 · 0,2021 · 0,2019 · 0,1966 · 0,1944.

### Por que esse resultado

O ajuste de hiperparâmetros rendeu uma queda de 0,0088 sobre o padrão (4,2%
relativo) — pequena frente à distância para o linear (0,056), o que confirma
que a maior parte da vantagem do boosting vem da família do modelo (árvores
em sequência capturando interação), não do ajuste fino. É consistente com o
padrão geral do projeto: a busca de hiperparâmetros afina, mas não substitui
a escolha certa de modelo.

### A hipótese se confirmou?

Sim: venceu com CV muito próxima de 0,20 (0,1988), e a vantagem sobre os
lineares (Ridge/OLS ≈ 0,2551) é grande e consistente — todas as 5 folds
favorecem o boosting, com diferença média de 0,0563, muito acima do limiar de
0,005 do critério de decisão (`docs/comparacao_modelos.md`, seção "Aplicação
do critério"). A ressalva de explicabilidade também se confirmou: é o
segundo colocado nessa dimensão, atrás só da árvore de decisão isolada.

### Limitação conhecida

O modelo depende de `venda_direta` e de outras features que só ele consegue
inferir sozinho a partir de `bairro × área_útil` — parte da vantagem medida
aqui contra o linear é engenharia de atributos implícita, não capacidade pura
do algoritmo (`docs/protocolo_comparacao.md` §4, `docs/modelagem.md` §9.11).
A interação `bairro × área_útil` ainda não está explícita como coluna na
matriz — ver a seção "Viés em aberto" de `docs/comparacao_modelos.md`.
