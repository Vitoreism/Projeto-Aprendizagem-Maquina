# -*- coding: utf-8 -*-
"""Testes da extração de campos estruturados a partir da descrição (issue #3).

O risco aqui não é o padrão deixar de casar — é ele casar com a coisa errada e
gravar um número inventado por cima de um campo que o portal deixou vazio.
"""

import numpy as np
import pandas as pd
import pytest

from imoveis_jp.processing import enrich_from_description as enr


def _extrair(texto: str, campo: str):
    return enr.extrair(enr.normalizar(texto), campo)


def test_extrai_os_quatro_campos():
    assert _extrair("Apartamento com 3 quartos", "quartos") == 3
    assert _extrair("3 Dormitórios, sendo 1 suíte", "suites") == 1
    assert _extrair("2 Banheiros e varanda", "banheiros") == 2
    assert _extrair("1 Vaga de garagem coberta", "garagens") == 1


def test_aceita_numero_por_extenso():
    # aparecem cerca de 900 vezes na base ("dois quartos", "três suítes")
    assert _extrair("imóvel com dois quartos amplos", "quartos") == 2
    assert _extrair("três suítes com closet", "suites") == 3


def test_ignora_acento_e_caixa():
    assert _extrair("3 DORMITÓRIOS", "quartos") == 3
    assert _extrair("2 sUíTeS", "suites") == 2


def test_nao_inventa_quando_o_texto_nao_diz():
    assert _extrair("excelente apartamento no Bessa", "quartos") is None
    assert _extrair("", "suites") is None
    assert enr.extrair(enr.normalizar(None), "quartos") is None


def test_descarta_valor_fora_da_faixa_plausivel():
    # número de telefone, CEP ou erro de digitação não viram 200 quartos
    assert _extrair("200 quartos", "quartos") is None
    assert _extrair("99 vagas", "garagens") is None


def test_distancia_em_metros_nao_vira_area():
    # 'area_util' não é extraída: as descrições citam várias áreas e nenhum
    # padrão passou de 67% de acerto contra o valor estruturado
    assert enr.CAMPO_NAO_EXTRAIDO == "area_util"
    assert "area_util" not in enr.EXTRATORES


def test_enriquecer_preenche_so_o_que_esta_ausente():
    df = pd.DataFrame(
        {
            "descricao_completa": [
                "Apartamento com 3 quartos e 2 vagas",
                "Cobertura com 4 quartos",
            ],
            "quartos": [2.0, np.nan],  # o primeiro já veio do portal
            "garagens": [np.nan, np.nan],
        }
    )
    resultado, preenchidos = enr.enriquecer(df)

    assert resultado["quartos"].tolist() == [2.0, 4.0]
    assert preenchidos["quartos"] == 1
    assert resultado["garagens"].tolist()[0] == 2.0


def test_enriquecer_sem_coluna_de_texto_nao_quebra():
    df = pd.DataFrame({"quartos": [np.nan]})
    resultado, preenchidos = enr.enriquecer(df)

    assert preenchidos == {}
    assert resultado["quartos"].isna().all()


def test_validar_mede_contra_o_campo_ja_preenchido():
    df = pd.DataFrame(
        {
            "descricao_completa": ["com 3 quartos", "com 2 quartos", "com 5 quartos"],
            "quartos": [3.0, 2.0, 4.0],  # o terceiro discorda do texto
        }
    )
    relatorio = enr.validar(df)
    linha = relatorio[relatorio["campo"] == "quartos"].iloc[0]

    assert linha["comparaveis"] == 3
    assert linha["exato"] == pytest.approx(2 / 3)
    assert linha["ate_1_de_diferenca"] == 1.0
