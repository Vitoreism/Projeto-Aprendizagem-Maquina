# -*- coding: utf-8 -*-
"""Smoke tests do layout do projeto.

Se estes quebrarem, é sinal de que alguma pasta foi movida sem atualizar
:mod:`imoveis_jp.config` — que é o único lugar onde caminhos são definidos.
"""

from imoveis_jp import config


def test_root_aponta_para_a_raiz_do_repo():
    assert (config.ROOT / "pyproject.toml").is_file()
    assert (config.ROOT / "src" / "imoveis_jp").is_dir()


def test_pastas_de_dados_seguem_o_layout():
    assert config.RAW == config.ROOT / "data" / "raw"
    assert config.INTERIM == config.ROOT / "data" / "interim"
    assert config.PROCESSED == config.ROOT / "data" / "processed"
    assert config.ANUNCIOS_JSON.parent == config.RAW


def test_links_cache_nomeia_pelo_escopo():
    p = config.links_cache("pb-joao-pessoa", "apartamento", "venda")
    assert p.parent == config.RAW
    assert p.name == "links_pb-joao-pessoa_apartamento_venda.json"


def test_ensure_dirs_e_idempotente():
    config.ensure_dirs()
    config.ensure_dirs()
    assert config.RAW.is_dir() and config.INTERIM.is_dir() and config.PROCESSED.is_dir()
