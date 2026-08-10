# Protocolo de comparação de modelos — Etapa 5

**Módulo:** `src/imoveis_jp/models/candidatos/`
**Issue:** #20 (Fase 0) · **Bloqueia:** #21, #22, #23, #24, #25

---

## 1. Como inscrever um modelo

Crie **um** arquivo no pacote e exporte **uma** constante. Não existe lista
central para editar — é por isso que cinco pessoas trabalham na mesma semana sem
colidir.

```python
# src/imoveis_jp/models/candidatos/arvore.py
from sklearn.tree import DecisionTreeRegressor

from imoveis_jp.models import dataset
from imoveis_jp.models.candidatos.base import Candidato

CANDIDATO = Candidato(
    nome="arvore_decisao",
    dono="dev A",
    regressor=DecisionTreeRegressor(random_state=dataset.SEMENTE),
    grade={"regressor__max_depth": [4, 8, 16, None]},
    hipotese=(
        "Perde do boosting porque uma árvore única não faz média sobre "
        "partições, mas é a única que entrega a regra em texto legível. "
        "Espera-se CV entre 0,24 e 0,28."
    ),
)
```

`descobrir()` acha o arquivo sozinho. O modelo entra automaticamente no
`train.py` e — se tiver `grade` — no `tune.py`.

Rode `pytest tests/test_candidatos.py` antes de commitar. **É o teste que
economiza o dia da consolidação:** se o modelo não aceita nulo, categoria em
texto ou o formato das colunas, ele quebra em dois segundos em vez de na Fase 2.

---

## 2. As quatro regras do protocolo

1. **Mesmo split, mesmas folds, mesmo `Pipeline`.** `GroupShuffleSplit` +
   `GroupKFold(5)` + `SEMENTE = 42`. Preparação própria invalida a comparação.
2. **Critério de decisão declarado antes dos resultados** (issue #25).
3. **O teste é tocado uma vez, por uma pessoa.** Cinco pessoas olhando o teste
   durante o desenvolvimento o transformam num segundo conjunto de validação.
4. **Hipótese registrada antes de rodar.**

As regras 1 e 4 são verificadas por teste. As regras 2 e 3 dependem de
disciplina — não há como um `assert` saber quando alguém olhou o teste.

### Por que a regra 4 virou código

O `Candidato` **recusa** hipótese vazia, e o teste recusa hipótese com menos de
40 caracteres. Não é burocracia: três dos achados mais úteis do projeto foram
previsões **erradas**, e nenhuma teria aparecido se a hipótese pudesse ser
escrita depois do resultado.

| previsão | resultado |
|---|---|
| corrigir o vazamento estrutural custaria acurácia | melhorou (0,2232 → 0,2155) |
| o erro se concentraria no alto padrão | as duas pontas erravam igual |
| `venda_direta` renderia 0,005 a 0,015 | rendeu 0,0010 — dez vezes menos |

Escrever a hipótese depois transforma qualquer resultado em confirmação.

---

## 3. As três decisões que a issue #20 deixou em aberto

### 3.1 Quem consome o registro — os dois

O `Candidato` carrega `regressor` **e** `grade` porque os dois fluxos leem dele:

| módulo | usa | ignora |
|---|---|---|
| `train.py` | `regressor`, `escalar_binarias` | `grade` |
| `tune.py` | `grade`, `escalar_binarias` | — |

`grade` vazia significa **"sem busca"**, e o `tune.py` pula o candidato
avisando. Não significa "esqueci" — quem esqueceu não passa no teste.

`train.py::montar_modelos` continua com **duas referências fixas**, que não são
candidatas e não deveriam ser:

- **`baseline_mediana`** é piso de sanidade, não modelo. Não tem hipótese a
  registrar nem grade a buscar; existe para provar que a montagem está correta
  (R² tem que dar ≈ 0).
- **`gradient_boosting`** na configuração padrão mede o ganho do ajuste.
  Transformá-lo em candidato apagaria a referência de quanto a busca rendeu.

### 3.2 `escalar_binarias` — `StandardScaler`, e o default é `False`

As contínuas saem do `StandardScaler` com variância 1. As binárias passam
direto, e uma binária com *p* = 0,1 tem variância 0,09. Para KNN e MLP isso faz
as contínuas dominarem a distância e o gradiente; para árvore e Ridge é
irrelevante — a árvore corta por limiar, e limiar em 0/1 não muda com a escala.

**Por que `StandardScaler` e não `MinMaxScaler`:** MinMax em dado que já é 0/1 é
literalmente a identidade. Deixaria a binária com variância 0,09 e as contínuas
com 1 — ou seja, não resolveria nada. Só a padronização iguala as variâncias,
que é o problema.

> **Um detalhe que ninguém tinha notado.** As colunas indicadoras de ausência
> geradas pelo `add_indicator` também são binárias, mas nascem **dentro** do
> bloco `num` e já passam pelo `StandardScaler` hoje, independentemente desta
> opção. A base já escalava metade das suas binárias e não escalava a outra
> metade, sem que isso tivesse sido decidido por ninguém. Com
> `escalar_binarias=True` o tratamento fica coerente; com `False`, a
> inconsistência permanece — e é inofensiva para os modelos que usam o default.

O default é `False` porque foi o comportamento sob o qual todos os números já
relatados foram medidos. Um teste trava isso (`test_o_default_nao_muda_o_pre_processamento_existente`).

### 3.3 Reprodutibilidade é exigida, não sugerida

O `Candidato` não tem campo para semente — mas o teste
`test_candidato_e_reprodutivel` verifica que, **se** o estimador aceita
`random_state`, ele está em `dataset.SEMENTE`.

Árvore e MLP sorteiam: inicialização de pesos, desempate de split, fatia do
early stopping. Sem semente fixa, dois devs rodando o mesmo candidato obtêm
números diferentes, e a Fase 2 não teria como saber se a diferença entre dois
modelos é o modelo ou o sorteio. A verificação é condicional porque nem todo
estimador é estocástico — exigir de todos seria exigir o impossível.

---

## 4. ⚠️ O viés que pode contaminar a comparação inteira

`venda_direta` (§9.11 de [modelagem.md](modelagem.md)) rendeu **0,0045 ao Ridge
e 0,0010 ao boosting**. A diferença não é ruído: é o pedaço que a árvore
reconstruía sozinha a partir de `bairro` × `area_util`, e que o modelo linear
não tem como inferir.

**Consequência para esta etapa:** toda feature que só o modelo não-linear
consegue inferir sozinho **enviesa a comparação a favor dele**. Se ela ficar
implícita, o resultado credita ao algoritmo um ponto que pertence à engenharia
de atributos.

Antes de fechar a Fase 2, vale varrer se há outras features nessa condição.
Interação área × bairro é a candidata óbvia e ainda não está explícita na
matriz.

---

## 5. Referência de base

| Modelo | CV MAE (log) | Teste MAE | Erro % mediano | R² (log) |
|---|---|---|---|---|
| Gradient Boosting ajustado | 0,1988 ± 0,0030 | R$ 170.150 | 15,6% | 0,897 |
| Gradient Boosting (padrão) | 0,2076 ± 0,0056 | R$ 172.892 | 16,1% | 0,892 |
| Ridge | 0,2551 ± 0,0022 | R$ 249.200 | 19,1% | 0,844 |
| Baseline (mediana) | 0,6183 ± 0,0024 | R$ 443.793 | 43,1% | −0,002 |

Base: 15.476 linhas, 76 features na matriz, 132 colunas depois do one-hot.

---

## 6. Execução

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_candidatos.py
.\.venv\Scripts\python.exe -m imoveis_jp.models.train
.\.venv\Scripts\python.exe -m imoveis_jp.models.tune
```
