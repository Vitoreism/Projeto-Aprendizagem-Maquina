# -*- coding: utf-8 -*-
"""Gradient Boosting ajustado -- o modelo de referencia da comparacao.

A configuracao abaixo e a vencedora de quatro passadas de GridSearchCV, e cada
valor tem historia. Ela esta fixada aqui, e nao lida de um json, para que o
treino seja reproduzivel sem depender de rodar a busca antes.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from imoveis_jp.models import dataset
from imoveis_jp.models.candidatos.base import Candidato

#: grade refinada em quatro passadas. O historico completo esta na secao 8 de
#: docs/modelagem.md; o resumo de cada eixo:
#:
#: max_leaf_nodes -- o unico eixo que mexeu de verdade no resultado. A primeira
#:   grade ia so ate 63 e o otimo caiu NA BORDA melhorando (15 -> 0,2346,
#:   31 -> 0,2291, 63 -> 0,2262), sinal classico de grade mal-posta. Estendida
#:   ate 255.
#:
#: max_iter -- valor fixo, teto e nao alvo. As duas melhores configuracoes com
#:   200 e com 500 ficaram a 0,00002 uma da outra porque early_stopping='auto'
#:   liga sozinho acima de 10.000 amostras (temos 12.214) e o modelo para antes
#:   do teto.
#:
#: min_samples_leaf -- caiu na borda tres vezes seguidas, cada uma depois de uma
#:   mudanca na base. Na quarta medicao (base sem os repasses) a curva finalmente
#:   sobe dos dois lados:
#:
#:      1 -> 0,1998   2 -> 0,1997   3 -> 0,1989   5 -> 0,1998
#:      8 -> 0,2006  10 -> 0,2022  15 -> 0,2024  20 -> 0,2033
#:
#:   Minimo interior em 3, sem grade a estender -- e nada a mudar, porque 1, 2,
#:   3 e 5 cabem dentro de 0,0009 uns dos outros e o limiar do projeto e 0,005.
#:
#: l2_regularization -- EIXO MORTO, fixado no default. A media entre
#:   configuracoes deu 0,2234 com l2=0 e 0,2235 com l2=1: diferenca na quarta
#:   casa. Mantido como valor unico para nao gastar metade das configuracoes
#:   com um eixo que nao responde.
GRADE = {
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__max_iter": [500],
    "regressor__max_leaf_nodes": [31, 63, 127, 255],
    "regressor__min_samples_leaf": [2, 3, 5, 10, 20, 50],
    "regressor__l2_regularization": [0.0],
}

#: por que min_samples_leaf=5 e nao 3, que e o minimo medido:
#:
#: A regra de decisao do projeto tem duas partes. A primeira diz que vence o
#: menor MAE da CV. A segunda diz quando se pode DECLARAR vantagem sobre o
#: segundo colocado: diferenca >= 0,005. Os 0,0009 entre 3 e 5 nao chegam la --
#: e regiao plana, nao diferenca. Desempatado pela folha maior, que e o ponto
#: mais conservador da faixa ao mesmo custo em CV.
#:
#: O valor anterior era 10, e veio de um argumento de dominio: preco de imovel
#: tem cauda longa, e folha maior impede a arvore de isolar um unico anuncio de
#: alto padrao e decorar o valor dele. A medicao contradisse esse prior nesta
#: faixa -- a media sobre as 8 configuracoes de cada nivel cai monotonicamente
#: (5 -> 0,2089, 10 -> 0,2109, 20 -> 0,2126, 50 -> 0,2187) -- e o prior cedeu.
CANDIDATO = Candidato(
    nome="gradient_boosting_ajustado",
    dono="dev E (integrador)",
    regressor=HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=127,
        min_samples_leaf=5,
        l2_regularization=0.0,
        random_state=dataset.SEMENTE,
    ),
    grade=GRADE,
    hipotese=(
        "E o favorito, e a comparacao existe justamente para nao aceitar isso "
        "sem prova. A vantagem esperada sobre os lineares vem de interacao "
        "(area x bairro) e de nao-linearidade na cauda de preco. Onde ele deve "
        "perder: explicabilidade contra a arvore unica, e extrapolacao acima de "
        "R$ 2 milhoes, onde ha pouco dado. Espera-se CV proxima de 0,20."
    ),
    # arvore nao usa distancia nem gradiente sobre a escala das features: ela
    # corta por limiar, e um limiar em 0/1 e o mesmo antes e depois de escalar.
    escalar_binarias=False,
)
