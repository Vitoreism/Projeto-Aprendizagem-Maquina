# -*- coding: utf-8 -*-
"""Dashboard de comparacao dos modelos (spec: docs/superpowers/specs/2026-08-10-dashboard-streamlit-design.md).

Nenhum modulo deste pacote importa Streamlit. O cache (`st.cache_data`,
`st.cache_resource`) mora no `app.py`, na raiz do repositorio, embrulhando as
funcoes daqui. E o que permite testar carregamento, previsao e figuras sem
subir servidor -- e o que impede que a logica do projeto fique presa dentro de
um framework de UI.
"""
