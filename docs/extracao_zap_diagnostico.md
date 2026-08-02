# Diagnóstico — por que a extração via LLM do ZapImóveis não produzia dados

**Módulo:** `src/imoveis_jp/processing/extract_llm_features.py`
**Data:** 02/08/2026

---

## 1. O sintoma

A base do zap tinha `posicao_solar` como "Nao informado" em **11.833 dos 11.841**
anúncios, `tipo_unidade` como "Apartamento tipo" em 100% deles e
`diferenciais_unicos` vazio em todos menos um. A conclusão natural — e errada —
era que a extração nunca tinha sido executada.

Ela foi. Gravou 11.841 registros vazios e **reportou sucesso**.

---

## 2. A causa: prompt e parser discordavam

O fluxo de lote funciona assim:

1. `extrair_lote_atributos_llm` monta o payload com um `id_lote` por imóvel;
2. o modelo responde com uma lista em `"resultados"`;
3. o parser casa resultado ↔ imóvel **pelo `id_lote` devolvido**.

O problema está no passo 3: `construir_prompt_dinamico_batch`, o prompt da etapa
2, **nunca pedia** que o modelo devolvesse o `id_lote`. Ele só aparece no prompt
da etapa 1 (`SYSTEM_PROMPT_DISCOVERY_BATCH`).

Sem o campo, `item_res.get("id_lote")` era `None` em todo resultado, a condição
`if id_lote is not None` nunca passava, `mapeamento_final` ficava vazio e o laço
seguinte preenchia cada URL com o default.

A resposta crua da API mostra que o modelo sempre soube fazer o trabalho:

```json
{"resultados": [{"posicao_solar": "Nascente",
                 "status_construcao": "Pronto para morar",
                 "piscina": true, "academia": true, ...}]}
```

Tudo lá, menos o `id_lote`.

**O que fez o problema durar não foi a falha, foi o silêncio.** Um lote inteiro
caindo no default é indistinguível, para quem lê o log, de um lote de imóveis
que genuinamente não tinham atributos.

### Correções

1. o prompt da etapa 2 passa a pedir `id_lote` explicitamente;
2. o parser usa a **posição na lista** como fallback — o modelo devolve os
   resultados na ordem da entrada, então exigir o campo era frágil sem
   necessidade;
3. lote inteiro em default virou **aviso explícito**.

Medido em 12 anúncios com descrição, antes e depois:

| | Antes | Depois |
|---|---|---|
| `posicao_solar` informada | 0/12 | 5/12 |
| `status_construcao` informado | 0/12 | 8/12 |
| Atributos `True` por anúncio | 0,0 | 10,8 de 45 |

---

## 3. Economia de chamadas

Anúncio sem descrição ia para a API com o texto trocado por
`"sem descricao disponivel"` e voltava com tudo `False` — o mesmo resultado do
default, ao custo de uma chamada. No zap isso é **78% da base**.

Agora esses casos são resolvidos localmente: **434 chamadas em vez de 1.974**.

---

## 4. O limite de cota é por organização

A extração parou em 952 dos 2.602 anúncios com descrição:

```
429 - Rate limit ... tokens per day (TPD): Limit 500000, Used 499765
```

O teto de 500 mil tokens/dia da conta gratuita da Groq é **por organização, não
por chave**. Ter duas chaves da mesma conta não dobra nada. Reduzir workers não
ajuda — não é limite de velocidade, é de volume diário.

### Onde os tokens vão

O prompt de sistema tem **990 tokens** e é reenviado a cada lote:

| Lote | Chamadas | Tokens totais | Overhead do prompt |
|---|---|---|---|
| 6 (atual) | 275 | 678k | **40%** |
| 12 | 138 | 542k | 25% |
| 20 | 83 | 488k | 17% |
| 30 | 55 | 460k | 12% |

Com lote 6, o restante **não cabe** na cota diária. Com lote 20, cabe.

Antes de mudar o padrão, porém, valeria testar a qualidade em lote maior: a
documentação da issue #9 afirma que lote 6 é o "ponto ideal de riqueza (98,5%)",
mas não há evidência registrada que sustente esse número.

---

## 5. Por que a extração parcial foi revertida

A extração parcial **não é uma amostra aleatória**. A cota acabou no meio da
fila, e a fila é a ordem do JSON — que é a ordem de ranking de busca do portal.

| | n | Preço mediano |
|---|---|---|
| Processados pela API | 952 | R$ 498.284 |
| Não processados | 1.650 | R$ 430.000 |

**Mann-Whitney p = 0,0021**, medianas 16% distantes. Ou seja, a extração cobriu
preferencialmente os anúncios mais caros.

O efeito prático é um artefato: entre os anúncios do zap, "tem atributo extraído
da descrição" passa a significar, aproximadamente, "está entre os mais caros". O
modelo pode aprender `com_piscina=True → preço alto` em parte por causa de onde
a cota acabou, não porque piscina valoriza imóvel.

É a mesma classe de problema que a auditoria de pré-processamento encontrou em
`bairro_preco_m2_medio`: uma feature que carrega informação do alvo por um
caminho que não é o mecanismo real.

### E não havia ganho para compensar

| | Sem LLM no zap | Com 36,6% do zap |
|---|---|---|
| CV MAE (log) | 0,2232 ± 0,0031 | 0,2238 ± 0,0043 |
| Teste MAE | R$ 166.660 | R$ 168.559 |
| Erro % mediano | 16,4% | 16,6% |

A diferença é menor que o desvio entre folds. Artefato conhecido sem ganho
mensurável não se mantém — os dados derivados foram revertidos ao estado
uniforme, sem LLM no zap.

**O checkpoint continua versionado** (exceção explícita no `.gitignore`), porque
representa cota já consumida e permite retomar barato. Ele existe para uma
eventual execução **completa**, não para o estado parcial.

⚠️ Como o checkpoint está presente, rodar
`extract_llm_features --dataset zap --merge` **reaplica o estado parcial**. Só
faça isso depois de completar os 2.602.

---

## 6. O teto que a extração nunca vai furar

Mesmo completa, a extração não resolveria o artefato de portal em `com_piscina`:

| | Chaves na mão | Zap (hoje) | Zap (extração completa, estimado) |
|---|---|---|---|
| `com_piscina` | 48,1% | 0,6% | ~12% |

A diferença real entre os portais nunca foi a extração. É que **78% dos anúncios
do zap não têm descrição nenhuma**. Nenhuma quantidade de LLM cria informação
que não está no texto.

Por isso `origem_anuncio_*` permanece na matriz como variável de controle, e
essa é a resposta estrutural — não a extração.
