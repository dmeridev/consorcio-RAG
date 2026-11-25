"""Tests de embeddings. Los que necesitan el modelo descargado van marcados como `modelo`."""

from __future__ import annotations

import numpy as np
import pytest

from rag_consorcio.corpus import CORPUS
from rag_consorcio.embeddings import MODELO_POR_DEFECTO, similitud
from rag_consorcio.extraer import extraer
from rag_consorcio.trocear import por_articulo


def test_similitud_es_el_producto_escalar_de_vectores_normalizados():
    consulta = np.array([1.0, 0.0])
    vectores = np.array([[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]])
    assert similitud(consulta, vectores) == pytest.approx([1.0, 0.0, np.sqrt(0.5)])


@pytest.mark.modelo
def test_ningun_fragmento_supera_los_512_tokens_del_modelo():
    """Si un fragmento pasa de 512 tokens, el modelo corta el final sin avisar."""
    from transformers import AutoTokenizer

    tokenizador = AutoTokenizer.from_pretrained(MODELO_POR_DEFECTO)
    for documento in CORPUS:
        for fragmento in por_articulo(extraer(documento)):
            tokens = len(tokenizador(f"passage: {fragmento.texto}")["input_ids"])
            assert tokens <= 512, f"{fragmento.id}: {tokens} tokens"


@pytest.mark.modelo
def test_busqueda_por_significado_encuentra_el_plazo_de_carencia():
    """La pregunta no contiene ninguna palabra del artículo y aun así lo encuentra."""
    from rag_consorcio.embeddings import E5Local

    modelo = E5Local()
    fragmentos = por_articulo(extraer(next(d for d in CORPUS if d.id == "reglamento")))
    vectores = modelo.pasajes([f.texto for f in fragmentos])
    puntuaciones = similitud(modelo.consulta("¿Hay carencia si cambio de compañía?"), vectores)
    mejores = [fragmentos[i].unidades[0] for i in np.argsort(-puntuaciones)[:3]]
    assert "reglamento:texto:art. 8" in mejores
