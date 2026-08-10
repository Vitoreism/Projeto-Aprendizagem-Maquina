# -*- coding: utf-8 -*-
"""Testes da canonizacao de bairro.

`bairro` e o atributo mais importante do modelo (+0,2226 de MAE na importancia
por permutacao, mais que o MAE total). Cada teste aqui trava um defeito real que
existia na versao anterior e que corrompia 14% dos anuncios.
"""

import pandas as pd
import pytest

from imoveis_jp.processing import deduplicate_dataset as dd


def test_nome_mais_especifico_vence_o_mais_generico():
    """O bug que mais custava: 511 anuncios do Altiplano viravam Cabo Branco.

    A lista antiga era percorrida na ordem em que foi escrita, e 'cabo branco'
    vinha antes de 'altiplano'. Sao bairros distintos, com preco/m2 mediano de
    R$ 15.000 e R$ 11.156 -- 26% de diferenca fundida numa categoria so.

    O bairro oficial se chama Altiplano, de um token so. Sozinho ele perderia
    de 'cabo branco', de dois tokens; por isso 'altiplano cabo branco' esta em
    ALIASES_BAIRRO e entra na ordenacao por especificidade com tres.
    """
    assert dd.casar_bairro("Altiplano Cabo Branco") == "altiplano"
    assert dd.casar_bairro("Altiplano") == "altiplano"
    assert dd.casar_bairro("Cabo Branco") == "cabo_branco"

    # o mesmo par, agora dentro de um endereco completo
    assert (
        dd.extrair_bairro("Rua Silvino Lopes, 50, Altiplano Cabo Branco,João Pessoa/PB")
        == "altiplano"
    )


def test_tambau_nao_engole_tambauzinho():
    # 119 anuncios, 31% de diferenca de preco/m2 entre os dois
    assert dd.casar_bairro("Tambauzinho") == "tambauzinho"
    assert dd.casar_bairro("Tambaú") == "tambau"


def test_nunca_inventa_bairro():
    """O fallback antigo devolvia a primeira palavra com mais de 3 letras.

    Produzia 'avenida', 'doutor', 'professor', 'maria' como se fossem bairros --
    1.589 anuncios (9,8%) em categorias que nao existem. 'avenida' juntava
    imoveis de toda a cidade: IQR de preco/m2 de R$ 6.898, contra R$ 1.654 de um
    bairro real.
    """
    for texto in ["Avenida", "Doutor", "Professor Fulano", "Rua Qualquer Coisa"]:
        assert dd.casar_bairro(texto) is None

    # endereco sem bairro reconhecivel nao vira categoria nova
    assert dd.extrair_bairro("Praia de Intermares, João Pessoa - PB") == "nao_informado"
    assert dd.extrair_bairro(None) == "nao_informado"
    assert dd.extrair_bairro("") == "nao_informado"


def test_le_os_dois_formatos_de_endereco():
    """Os portais escrevem diferente, e a quebra precisa cobrir os dois."""
    # chaves na mao: campo separado por virgula
    assert (
        dd.extrair_bairro("Avenida Governador Argemiro De Figueiredo, 155, Jardim Oceania,João Pessoa/PB")
        == "jardim_oceania"
    )
    # zapimoveis: bairro depois de hifen
    assert (
        dd.extrair_bairro("Rua Ambrosina Soares dos Santos, 38 - Bessa, João Pessoa - PB")
        == "bessa"
    )
    # endereco que e so o bairro
    assert dd.extrair_bairro("Miramar,João Pessoa/PB") == "miramar"


def test_campo_do_portal_tem_precedencia():
    # o campo 'bairro' do zap acerta 99,9% (9.260 de 9.273); o endereco e o
    # plano B, e o unico caminho para os 6.473 anuncios do chaves na mao
    assert dd.extrair_bairro("Rua Qualquer, 10, João Pessoa/PB", "Bessa") == "bessa"
    # campo do portal invalido nao impede o endereco de resolver
    assert dd.extrair_bairro("Miramar,João Pessoa/PB", "João Pessoa") == "miramar"


def test_artigo_nao_distingue_bairro():
    # 'Valentina Figueiredo' e 'Valentina de Figueiredo' sao o mesmo lugar
    assert dd.casar_bairro("Valentina Figueiredo") == "valentina_de_figueiredo"
    assert dd.casar_bairro("Valentina de Figueiredo") == "valentina_de_figueiredo"
    # 'Ponta dos Seixas' contra o oficial 'Ponta Do Seixas'
    assert dd.casar_bairro("Ponta dos Seixas") == "ponta_do_seixas"


def test_apelido_designa_o_mesmo_bairro_oficial():
    """Cada apelido e uma afirmacao sobre o mesmo lugar, nao um chute."""
    # a Wikipedia lista 'Varjao (Rangel)': dois nomes, um bairro
    assert dd.casar_bairro("Rangel") == "varjao"
    assert dd.casar_bairro("Varjão") == "varjao"
    # variantes de escrita do nome oficial
    assert dd.casar_bairro("Planalto Boa Esperança") == "planalto_da_boa_esperanca"
    assert dd.casar_bairro("José Américo de Almeida") == "jose_americo"
    assert dd.casar_bairro("Jardim 13 de Maio") == "treze_de_maio"


def test_praia_de_outro_municipio_nao_vira_bairro():
    """Chutar o bairro mais proximo repetiria o defeito com outra roupa."""
    assert dd.casar_bairro("Praia de Camboinha") is None   # Cabedelo
    assert dd.casar_bairro("Praia de Carapibus") is None   # Conde
    assert dd.casar_bairro("Praia de Intermares") is None  # Cabedelo


def test_localidade_reconhecida_e_categoria_propria():
    """Nao esta entre os 64 oficiais, mas funciona como bairro no mercado.

    Jardim Luna tem 41 anuncios com CV de preco/m2 de 0,23 -- mais homogeneo
    que o bairro oficial mediano (0,35) e bem mais caro que a cidade
    (R$ 10.132/m2 contra R$ 6.082). Manda-lo para 'nao_informado' descartaria
    sinal bom.
    """
    assert dd.casar_bairro("Jardim Luna") == "jardim_luna"
    assert dd.casar_bairro("Novo Milênio") == "novo_milenio"
    assert dd.casar_bairro("Colinas do Sul") == "colinas_do_sul"

    # a lista continua fechada e curada: nao virou fallback
    assert dd.casar_bairro("Jardim Qualquer") is None
    assert dd.casar_bairro("Conjunto Inexistente") is None


def test_apelido_aponta_para_bairro_existente():
    """Guarda contra erro de digitacao no mapa de apelidos."""
    assert not (set(dd.ALIASES_BAIRRO.values()) - dd.CANONICOS)


def test_lista_oficial_tem_os_64_bairros():
    assert len(dd.BAIRROS_OFICIAIS) == 64
    # os oficiais mais as localidades reconhecidas
    assert len(dd.CANONICOS) == 64 + len(dd.LOCALIDADES_RECONHECIDAS)


def test_numero_no_campo_nao_atrapalha():
    # '38 - Bessa' e '230 - Cabo Branco' aparecem na base
    assert dd.casar_bairro("38 Bessa") == "bessa"
    assert dd.casar_bairro("230 - Cabo Branco") == "cabo_branco"


def test_cidade_e_estado_nunca_viram_bairro():
    for texto in ["João Pessoa", "PB", "Paraíba"]:
        assert dd.casar_bairro(texto) is None


def test_joao_pessoa_no_filtro_nao_derruba_bairros_com_joao():
    """O filtro da cidade e por par exato, nao por palavra solta."""
    assert dd.casar_bairro("João Paulo II") == "joao_paulo_ii"
    assert dd.casar_bairro("João Agripino") == "joao_agripino"


def test_toda_saida_e_nome_oficial():
    """A garantia central: ou e bairro da lista, ou e 'nao_informado'."""
    oficiais = dd.CANONICOS | {"nao_informado"}
    entradas = [
        "Rua X, 1, Manaíra,João Pessoa/PB",
        "Rua Y, 2 - Gramame, João Pessoa - PB",
        "Endereço totalmente sem bairro",
        None,
        float("nan"),
    ]
    for e in entradas:
        assert dd.extrair_bairro(e) in oficiais


def test_base_real_so_tem_bairro_oficial():
    """Roda sobre a base inteira: nenhum valor fora da lista."""
    caminho = dd.config.PROCESSED / "features_matrix.csv"
    if not caminho.exists():
        pytest.skip("matriz de features ainda nao gerada")

    bairros = set(pd.read_csv(caminho, low_memory=False)["bairro"].dropna())
    oficiais = dd.CANONICOS | {"nao_informado"}

    assert not (bairros - oficiais), f"bairros fora da lista: {sorted(bairros - oficiais)}"
