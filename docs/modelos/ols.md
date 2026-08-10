> Documentação escrita durante a consolidação da Fase 2 (issue #25) porque não
> havia sido entregue com o candidato (issue #24, fechada sem o markdown).
> Números tirados de `data/processed/resultados_modelos.csv` e
> `cv_mae_por_fold.csv`, gerados por esta mesma issue — nenhum valor inventado.

## OLS (Regressão Linear Ordinária)

**Dono:** dev OLS · **Arquivo:** `src/imoveis_jp/models/candidatos/ols.py`

### O que este modelo assume sobre os dados

`LinearRegression` sem regularização: mesma suposição aditiva do Ridge, sem
penalidade sobre os coeficientes. Serve de referência para isolar o efeito da
regularização — se OLS e Ridge derem resultados parecidos, o gargalo do
modelo linear não é variância de coeficiente, é forma funcional.

### Hipótese registrada ANTES de rodar

> "Deve ficar muito perto do Ridge (CV MAE(log) ~0,25), porque a busca de
> alpha do Ridge não melhorou o default — sinal de que o gargalo do linear
> não é regularização, e forma funcional. Espera-se perder do boosting pela
> mesma margem do Ridge: preço depende de interações (área x bairro) que o
> modelo aditivo não representa. Se OLS for bem pior que Ridge, a
> colinearidade do one-hot está inflando os coeficientes."

### Grade testada, e por que essa faixa

```
GRADE = {}
```

Vazia de propósito: OLS não tem hiperparâmetro útil para buscar
(`fit_intercept` fica fixo em `True`, e o `Pipeline` já padroniza as contínuas
e faz o one-hot completo das nominais). Grade vazia significa "sem busca", não
"esqueceu" — é o próprio protocolo que define essa convenção.

### Resultado

| métrica | valor |
|---|---|
| CV MAE(log), média ± desvio | 0,2551 ± 0,0022 |
| Teste MAE (R$) | R$ 249.118 |
| Teste erro % mediano | 19,1% |
| Teste R² (log) | 0,844 |

Folds individuais (MAE log): 0,2574 · 0,2581 · 0,2536 · 0,2535 · 0,2531.

### Por que esse resultado

OLS e Ridge são, na prática, o mesmo modelo aqui: a diferença de CV é
0,00001 (0,25513 vs 0,25512), dentro do ruído entre folds. Isso é exatamente
a confirmação prevista — a colinearidade do one-hot **não** está inflando os
coeficientes o suficiente para a regularização fazer diferença mensurável, e
o teto de desempenho do linear vem de não representar interação
`área × bairro`, não de variância de coeficiente.

### A hipótese se confirmou?

Sim, nos dois pontos: ficou virtualmente empatado com Ridge (a hipótese
previu "muito perto"), e a hipótese alternativa ("se OLS for bem pior, é
colinearidade inflando coeficientes") foi corretamente descartada pelo
resultado — não há sinal de instabilidade por colinearidade.

### Limitação conhecida

Sem regularização, OLS depende inteiramente da matriz de atributos estar bem
condicionada — funciona aqui porque o `OneHotEncoder` já agrupa categorias
raras (`min_frequency=30`) antes de chegar ao regressor, evitando colunas
quase-duplicadas. Um dataset com categorias mais esparsas exporia essa
fragilidade de um jeito que este experimento não testa.
