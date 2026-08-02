# -*- coding: utf-8 -*-
"""Testes do módulo de extração via LLM.

Cobrem os contratos que já quebraram na prática: o formato do ranking entre a
etapa de descoberta e quem o lê, e o critério de descrição utilizável que decide
se um anúncio custa uma chamada de API.

A chamada à API em si não é testada — exigiria credencial e cota. O que dá para
travar sem rede é o que efetivamente falhou antes.
"""

import json

from imoveis_jp import config
from imoveis_jp.processing import extract_llm_features as ex


def test_descoberta_sem_descricao_nao_quebra():
    # antes esta função nem existia: --discover levantava NameError
    resultado = ex.executar_descoberta_amostral(
        clientes=[], imoveis=[{"url_anuncio": "u1", "descricao_completa": ""}]
    )

    assert len(resultado) == 0


def test_ranking_gravado_e_o_ranking_lido(tmp_path, monkeypatch):
    """O contrato entre a descoberta e carregar_atributos_do_ranking.

    A descoberta grava `ranking_frequencia`; o leitor corta nos 45 primeiros e
    exige termo com mais de 2 caracteres. Se os dois lados discordarem, o schema
    da extração silenciosamente vira a lista fixa de fallback.
    """
    monkeypatch.setattr(config, "INTERIM", tmp_path)

    (tmp_path / "discovered_attributes_rank.json").write_text(
        json.dumps(
            {
                "ranking_frequencia": {
                    "posicao solar nascente": 120,
                    "aceita fgts": 90,
                    "ok": 80,  # curto demais: o leitor descarta
                    "vista para o mar": 4,  # abaixo da frequência mínima
                }
            }
        ),
        encoding="utf-8",
    )

    atributos = ex.carregar_atributos_do_ranking(min_frequencia=5)

    assert atributos == ["posicao solar nascente", "aceita fgts"]


def test_sem_arquivo_de_ranking_cai_na_lista_fixa(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTERIM", tmp_path)

    atributos = ex.carregar_atributos_do_ranking()

    assert "piscina" in atributos
    assert len(atributos) == 45


def test_descricao_util_decide_quem_custa_chamada_de_api():
    # 78% dos anúncios do zap caem aqui; mandá-los para a API devolvia
    # exatamente o default, ao custo de uma chamada cada
    assert ex._tem_descricao_util({"descricao_completa": "Apartamento no Bessa com vista"})
    assert not ex._tem_descricao_util({"descricao_completa": "Descrição não encontrada."})
    assert not ex._tem_descricao_util({"descricao_completa": "   "})
    assert not ex._tem_descricao_util({"descricao_completa": None})
    assert not ex._tem_descricao_util({})


def test_prompt_da_etapa_2_pede_id_lote():
    # sem isso o parser não casa resultado com imóvel e TODO resultado vira
    # default em silêncio — foi o que zerou a extração inteira do zap
    prompt = ex.construir_prompt_dinamico_batch(["piscina", "academia"])

    assert "id_lote" in prompt
