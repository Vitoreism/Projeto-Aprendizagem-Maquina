# Dashboard Streamlit — comparação visual dos modelos

**Data:** 2026-08-10 · **Depende de:** issue #25 (artefatos da rodada única)

## Objetivo

Um app Streamlit que sirva a dois usos ao mesmo tempo: contar a história
metodológica do projeto para a banca (critério declarado antes → comparação
pareada → vencedor → só então o teste) e deixar o time cavar os resultados
(resíduos, importância, t-SNE, previsão de um imóvel).

Somente leitura dos artefatos já gerados pela issue #25. Nada de re-treinar a
comparação ao vivo: o MLP sozinho levou 20 minutos, e a issue exige que os
números venham de uma rodada única.

---

## Arquitetura

```
src/imoveis_jp/dashboard/
    __init__.py
    dados.py      carrega e valida os artefatos (via imoveis_jp.config)
    modelo.py     ajusta o vencedor, prevê, calcula a faixa de incerteza
    graficos.py   funções puras: recebem DataFrame, devolvem figura Plotly
app.py            entry point Streamlit: layout, texto e cache. Sem lógica.
tests/test_dashboard.py
```

**Regra de dependência que sustenta os testes:** `dados.py`, `modelo.py` e
`graficos.py` **não importam Streamlit**. São pandas/plotly/sklearn puros. Todo
o cache (`st.cache_data`, `st.cache_resource`) é aplicado em `app.py`,
embrulhando as funções do pacote. Consequência: a suíte de testes roda sem
Streamlit instalado e sem subir servidor.

Isso também respeita a regra nº 1 do README — nenhum caminho relativo ao
diretório atual; tudo sai de `imoveis_jp.config`.

### Dependências novas

`streamlit` e `plotly` em `requirements.txt`. Plotly e não Altair (que já vem
com o Streamlit) por causa do t-SNE: são 15.301 pontos, acima do limite padrão
de 5.000 linhas do Altair, e o Plotly renderiza via WebGL (`scattergl`) com
zoom fluido.

---

## Artefatos consumidos

| arquivo | usado em |
|---|---|
| `data/processed/resultados_modelos.csv` | veredito, teste |
| `data/processed/cv_mae_por_fold.csv` | comparação por fold |
| `data/processed/decisao_criterio.json` | veredito |
| `data/processed/resultados_pca.csv` | aba PCA |
| `data/processed/residuos_teste.csv` | resíduos, faixa de incerteza, sorteio |
| `data/processed/residuos_por_segmento.csv` | resíduos por bairro/faixa |
| `data/processed/importancia_permutacao.csv` | importância |
| `data/processed/importancia_permutacao_codificada.csv` | correlação × importância |
| `data/processed/tsne_coords.csv` | aba t-SNE |
| `data/processed/features_matrix.csv` | sorteio (para recuperar `url_anuncio`) |

Artefato ausente não derruba o app: a aba mostra qual comando o regenera
(ex.: `python -m imoveis_jp.models.train`) em vez de estourar traceback.

---

## Abas

### 1. O veredito (narrativa para a banca)

Auto-contida, na ordem em que a decisão realmente aconteceu:

1. O critério, declarado **antes** de qualquer resultado (as 4 regras, e por que
   o limiar é 0,005).
2. Ranking por CV MAE(log) dos seis candidatos.
3. A comparação pareada entre 1º e 2º, fold a fold.
4. O resultado: vantagem declarada para `gradient_boosting_ajustado`
   (5/5 folds, diferença média 0,0563).
5. O teste atrás de um botão **"revelar avaliação no teste"**.

O botão não é firula: encena a regra metodológica central da issue #25 — o
teste só é olhado depois da decisão fechada. Quem apresenta pode mostrar a
decisão, argumentar, e só então revelar.

### 2. Comparação por fold

Os seis candidatos nas mesmas cinco partições. Gráfico de linhas/pontos
(modelo × fold), mais a tabela com média e desvio. É o que torna a condição
"todas as folds a favor" verificável a olho.

### 3. PCA

Gráfico *dumbbell*: para cada modelo, um ponto sem PCA e outro com PCA, ligados
por uma linha, deixando a piora óbvia. Mais a hipótese registrada antes de rodar
e o veredito (piorou nos seis, sem exceção).

### 4. Resíduos

Onde o modelo erra, a partir de `residuos_teste.csv` e
`residuos_por_segmento.csv`: previsto × real em log-log, distribuição do
resíduo, erro por faixa de preço, e erro por bairro (só segmentos com n ≥ 30,
como já faz o `analysis.py`).

### 5. Importância

Importância por permutação (atributo inteiro), e o confronto correlação ×
importância — onde os dois discordam está o achado mais interessante do
projeto (`com_closet` com correlação 0,178 e importância zero; `bairro_bessa`
com correlação 0,010 e uma das dummies mais úteis).

### 6. t-SNE

Dispersão 2D colorida por faixa de preço, com aviso explícito de que é EDA e
não entrou em modelo nenhum.

### 7. Prever

Duas modalidades num seletor:

**(a) Preencher manualmente.** São 76 features — pedir todas seria absurdo. Os
campos expostos saem da importância por permutação já medida, não de palpite:
`area_util`, `bairro`, `garagens`, `quartos`, `suites`, `banheiros`,
`area_total`, `tipo_unidade`, `status_construcao`, `origem_anuncio`,
`venda_direta`. Num expander, as cinco comodidades com importância acima do
ruído: `com_piso_ceramica`, `com_varanda_gourmet`, `com_elevador`,
`com_vista_ou_acesso_praia`, `com_piso_porcelanato`. As outras 57 binárias
(63 no total, menos `venda_direta` que é campo principal e as 5 do expander)
vão como `False` — 34 dos 76 atributos têm importância indistinguível de zero.

A quinta nominal, `posicao_solar`, não é exposta: vai fixa em `nao_informado`,
que é a moda da base. Expor um campo que o anunciante quase nunca preenche
convidaria o usuário a inventar um valor que o modelo mal viu.

`condominio` e `iptu` ficam vazios por padrão e são enviados como **nulo**, não
como a mediana. O pipeline treinou com `SimpleImputer(add_indicator=True)`:
"iptu ausente" é sinal que o modelo aprendeu a usar (falta em ~80% dos
anúncios). Preencher com a mediana mentiria para o modelo; mandar nulo usa
exatamente o caminho em que ele foi treinado.

**(b) Sortear um imóvel real do teste.** Sorteia um dos 3.087 anúncios do
conjunto de teste — imóveis que o modelo nunca viu no treino — e mostra lado a
lado: o que o modelo previu, quanto custava de verdade, o erro, e o link para o
anúncio original (`url_anuncio`, recuperado de `features_matrix.csv` pelo
índice preservado em `residuos_teste.csv`).

#### De onde vem o modelo

Nada é persistido hoje: `train.py` treina e descarta. O app **reajusta o
vencedor na inicialização**, sob `st.cache_resource` — o fit levou 10,1s na
rodada da issue #25, então é custo único ao abrir. A alternativa (salvar um
`.joblib` no repositório) seria instantânea, mas cria um binário que pode
divergir em silêncio do código que o gerou. Dez segundos uma vez valem mais que
um artefato de procedência duvidosa.

#### Como a previsão é apresentada

Nunca um número seco. O erro mediano do modelo no teste é 15,6%, então
"R$ 487.234" seria precisão falsa. O app mostra a estimativa central e uma faixa
tirada da distribuição real de resíduos do teste (n = 3.087): os quantis 10% e
90% do erro percentual são **−27,1%** e **+36,9%**.

Dois avisos de extrapolação, ambos com motivo registrado:

- Área fora da faixa vista no treino (`area_util` entre 15 e 1.260 m²).
- Previsão acima de R$ 2 milhões — só 171 dos 3.087 anúncios de teste (5,5%)
  estão nessa faixa, e a hipótese registrada do boosting já previa que ele
  extrapola mal ali por falta de dado.

---

## Testes

`tests/test_dashboard.py`, sem subir servidor e sem exigir Streamlit:

- `dados.py`: carrega cada artefato com o schema esperado; arquivo ausente
  levanta erro com a mensagem do comando que o regenera.
- `modelo.py`: previsão devolve finito para uma entrada mínima; nulo em
  `condominio`/`iptu` não quebra (vai pelo imputer); faixa de incerteza é
  monótona (inferior < central < superior); avisos de extrapolação disparam nas
  bordas corretas.
- `graficos.py`: cada função devolve uma figura Plotly e sobrevive a DataFrame
  vazio.

---

## Fora de escopo

- Re-treinar a comparação pelo app (rodada única é requisito da issue #25).
- Colar link de anúncio (OLX ou outro): exigiria parser novo, esbarra no
  anti-bot, consome cota de LLM para extrair comodidades, e — o problema real —
  `origem_anuncio` é feature treinada com apenas três valores; um anúncio de
  outro portal não tem valor honesto para ela.
- Deploy remoto. O app roda local (`streamlit run app.py`, porta 8501).
