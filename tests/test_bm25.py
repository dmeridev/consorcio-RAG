"""Tests de BM25. El principal comprueba que nuestra implementación da las mismas
puntuaciones que una librería conocida (bm25s, variante «lucene») sobre el corpus real."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag_consorcio.bm25 import BM25, Tokenizador
from rag_consorcio.corpus import DIR_FRAGMENTOS


def test_tokenizador_minusculas_sin_tildes_ni_palabras_vacias():
    assert Tokenizador()("¿Qué dice el Artículo 9 del Reglamento?") == [
        "dice",
        "articulo",
        "9",
        "reglamento",
    ]


def test_tokenizador_conserva_la_ene():
    assert Tokenizador()("Daños en España") == ["daños", "españa"]


def test_tokenizador_con_raices_junta_variantes():
    tok = Tokenizador(raices=True)
    assert tok("asegurado") == tok("asegurados") == tok("asegurada")


def test_palabra_rara_pesa_mas_que_palabra_comun():
    docs = [["seguro", "franquicia"], ["seguro"], ["seguro"], ["seguro", "daños"]]
    bm25 = BM25(docs)
    assert bm25.idf["franquicia"] > bm25.idf["seguro"]


def test_sin_palabras_en_comun_puntua_cero():
    bm25 = BM25([["seguro"], ["daños"]])
    assert bm25.puntuar(["aerolitos"]).tolist() == [0.0, 0.0]


@pytest.fixture(scope="module")
def corpus_tokenizado():
    with open(DIR_FRAGMENTOS / "articulo.jsonl", encoding="utf-8") as f:
        textos = [json.loads(linea)["texto"] for linea in f]
    tok = Tokenizador()
    return [tok(t) for t in textos], tok


@pytest.mark.parametrize(
    "pregunta",
    [
        "¿Qué dice el artículo 9 del Reglamento?",
        "¿Hay carencia si cambio de compañía?",
        "daños por terrorismo y tumulto popular",
    ],
)
def test_coincide_con_bm25s(corpus_tokenizado, pregunta):
    bm25s = pytest.importorskip("bm25s")
    documentos, tok = corpus_tokenizado

    nuestro = BM25(documentos).puntuar(tok(pregunta))

    referencia = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    referencia.index(documentos, show_progress=False)
    esperado = referencia.get_scores(tok(pregunta))

    # bm25s («lucene») omite el factor (k1 + 1) del numerador, como hace Lucene desde la
    # versión 8: es constante, así que no cambia el orden de los resultados. Nuestras
    # puntuaciones son las suyas multiplicadas por k1 + 1 = 2,5.
    np.testing.assert_allclose(nuestro, esperado * 2.5, rtol=1e-4, atol=1e-6)
    assert np.argsort(-nuestro)[:10].tolist() == np.argsort(-esperado)[:10].tolist()
