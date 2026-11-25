"""Tests de la extracción y de las dos estrategias de troceado, sobre los PDFs reales."""

from __future__ import annotations

import itertools

import pytest

from rag_consorcio.corpus import CORPUS
from rag_consorcio.extraer import extraer
from rag_consorcio.numeros import normalizar_numero
from rag_consorcio.trocear import (
    MAX_CARACTERES,
    SOLAPE,
    TAMANO,
    _grupos,
    fijo,
    por_articulo,
)

DOCS = {d.id: d for d in CORPUS}


@pytest.fixture(scope="module")
def reglamento():
    return extraer(DOCS["reglamento"])


@pytest.fixture(scope="module")
def lcs():
    return extraer(DOCS["lcs"])


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("primero", "1"),
        ("diez", "10"),
        ("treinta y tres a)", "33 a)"),
        ("setenta y seis g)", "76 g)"),
        ("ciento", "100"),
        ("ciento seis quáter", "106 quáter"),
        ("8", "8"),
        ("único", "único"),
    ],
)
def test_normalizar_numero(texto, esperado):
    assert normalizar_numero(texto) == esperado


def test_el_indice_y_los_pies_de_pagina_no_entran(reglamento):
    assert ". . . ." not in reglamento.texto
    assert "BOLETÍN OFICIAL DEL ESTADO" not in reglamento.texto
    assert "LEGISLACIÓN CONSOLIDADA" not in reglamento.texto


def test_reglamento_separa_norma_aprobatoria_y_texto(reglamento):
    ids = {u.id for u in reglamento.unidades}
    assert "reglamento:norma:art. único" in ids
    assert {f"reglamento:texto:art. {n}" for n in range(1, 15)} <= ids


def test_ley_de_contrato_de_seguro_numera_con_cifras(lcs):
    ids = {u.id for u in lcs.unidades}
    assert "lcs:texto:art. 1" in ids
    assert "lcs:texto:art. 106 quáter" in ids
    # Los derogados no se indexan.
    assert "lcs:texto:art. 75" not in ids


def test_contexto_acumula_titulo_y_seccion(lcs):
    art1 = next(u for u in lcs.unidades if u.id == "lcs:texto:art. 1")
    assert art1.contexto.startswith("TÍTULO I")
    assert "Sección primera" in art1.contexto


def test_fragmento_por_articulo_lleva_cabecera(reglamento):
    art8 = next(f for f in por_articulo(reglamento) if f.id == "reglamento:texto:art. 8#1")
    assert art8.texto.startswith(
        "Reglamento del seguro de riesgos extraordinarios, art. 8. Plazo de carencia"
    )
    assert "siete días naturales" in art8.texto
    assert art8.unidades == ["reglamento:texto:art. 8"]


def test_ningun_fragmento_por_articulo_supera_el_maximo():
    for documento in CORPUS:
        for fragmento in por_articulo(extraer(documento)):
            assert len(fragmento.texto) <= MAX_CARACTERES, fragmento.id


def test_grupos_corta_por_apartados_sin_partir_parrafos():
    parrafos = ["1. " + "a" * 50, "b" * 30, "2. " + "c" * 50, "3. " + "d" * 50]
    grupos = _grupos(parrafos, maximo=90)
    assert grupos == [[parrafos[0], parrafos[1]], [parrafos[2]], [parrafos[3]]]


def test_fijo_solapa_y_cubre_todo_el_texto(reglamento):
    fragmentos = fijo(reglamento)
    assert fragmentos[0].inicio == 0
    assert fragmentos[-1].fin == len(reglamento.texto)
    for anterior, siguiente in itertools.pairwise(fragmentos):
        # Cada fragmento empieza SOLAPE caracteres antes de que acabe el anterior.
        assert anterior.fin - siguiente.inicio == SOLAPE
        assert len(anterior.texto) <= TAMANO
