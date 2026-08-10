> Documentação escrita durante a consolidação da Fase 2 (issue #25), pelo dono do
> candidato (`dono: "dev E (integrador)"` em `candidatos/ridge.py`) — não havia
> sido entregue junto com o registro do candidato.

## Ridge

**Dono:** dev E (integrador) · **Arquivo:** `src/imoveis_jp/models/candidatos/ridge.py`

### O que este modelo assume sobre os dados

Regressão linear com penalidade L2 sobre os coeficientes: assume que o log do
preço é uma combinação **aditiva** dos atributos (cada atributo contribui de
forma independente, sem interação), e usa a penalidade para conter a variância
dos coeficientes em face da colinearidade que o one-hot de 66 bairros introduz.

### Hipótese registrada ANTES de rodar

> "Perde do boosting por uma margem grande, e a margem é o resultado: ela mede
> quanta interação existe no problema. Preço de imóvel depende de área
> COMBINADA com bairro — 100 m² no Cabo Branco e 100 m² no Gramame não são o
> mesmo produto — e o modelo aditivo não representa isso. Espera-se CV entre
> 0,25 e 0,27, contra ~0,20 do boosting."

### Grade testada, e por que essa faixa

```
regressor__alpha: [0.1, 1.0, 10.0, 100.0, 1000.0]
```

Varredura logarítmica ampla — com as features padronizadas, o `alpha` útil
pode estar a ordens de magnitude do default, e passos menores não mudariam a
escolha. `escalar_binarias=False`: a penalidade de Ridge é sobre o coeficiente,
e o coeficiente de uma binária em 0/1 já está na escala do alvo; escalar
mudaria a força relativa da regularização entre binárias e contínuas sem
motivo para preferir uma das duas.

### Resultado

| métrica | valor |
|---|---|
| CV MAE(log), média ± desvio | 0,2551 ± 0,0022 |
| Teste MAE (R$) | R$ 249.200 |
| Teste erro % mediano | 19,1% |
| Teste R² (log) | 0,844 |

Folds individuais (MAE log): 0,2575 · 0,2581 · 0,2536 · 0,2535 · 0,2530 — desvio
entre folds pequeno (0,0022), a mesma ordem de grandeza do próprio limiar de
vantagem do projeto (0,005).

### Por que esse resultado

Fica dentro da faixa prevista, mas na ponta de baixo (0,2551, não 0,26+): a
busca de `alpha` (`melhores_hiperparametros.json`) devolveu o próprio default
1,0 como melhor configuração — sinal de que o gargalo aqui não é
regularização insuficiente, é forma funcional. Ridge e OLS ficam
estatisticamente indistinguíveis entre si (0,25512 vs 0,25513 — diferença na
quinta casa), confirmando que a colinearidade do one-hot não estava
prejudicando o OLS a ponto de a regularização ajudar.

### A hipótese se confirmou?

Sim, mas por baixo: previu-se 0,25–0,27 contra ~0,20 do boosting; medido foi
0,2551 contra 0,1988 — dentro da faixa prevista, e a diferença (0,0563) é
justamente a "margem grande" antecipada. O critério de decisão da issue #25
(seção "Aplicação do critério" em `docs/comparacao_modelos.md`) declara essa
vantagem formalmente: todas as 5 folds favorecem o boosting, com diferença
média de 0,0563 ≫ 0,005.

### Limitação conhecida

Ridge é o modelo mais explicável do grupo depois da árvore — cada coeficiente
tem leitura direta — mas essa explicabilidade não compensa o erro ~35% maior
em reais. Como nota o protocolo (`docs/protocolo_comparacao.md` §4), parte do
gap medido aqui é engenharia de atributos que falta: sem uma interação
explícita `área × bairro`, o modelo aditivo não tem como recuperar o que a
árvore reconstrói sozinha.
