"""Tests de las métricas de evaluación y del conjunto de preguntas."""

from __future__ import annotations

import pytest

from rag_consorcio.corpus import CORPUS
from rag_consorcio.evaluar import cargar_preguntas, metricas, posicion_acierto
from rag_consorcio.extraer import extraer


def test_posicion_acierto_admite_varios_articulos_validos():
    resultados = [["a:1"], ["b:2", "b:3"], ["c:4"]]
    assert posicion_acierto(resultados, ["b:3", "z:9"]) == 2
    assert posicion_acierto(resultados, ["z:9"]) is None


def test_metricas():
    # Aciertos en las posiciones 1, 2 y 4, y un fallo.
    m = metricas([1, 2, 4, None])
    assert m["recall@1"] == 0.25
    assert m["recall@3"] == 0.5
    assert m["recall@5"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 4)


def test_las_preguntas_citan_articulos_que_existen():
    existentes = {u.id for d in CORPUS for u in extraer(d).unidades}
    for pregunta in cargar_preguntas():
        for articulo in pregunta.articulos_validos:
            assert articulo in existentes, f"{pregunta.id}: {articulo}"


def test_ids_de_preguntas_unicos_y_tipos_conocidos():
    preguntas = cargar_preguntas()
    assert len({p.id for p in preguntas}) == len(preguntas)
    tipos = {"parafrasis", "literal", "caso", "referencia", "sin_respuesta"}
    assert {p.tipo for p in preguntas} <= tipos
    assert all(not p.articulos_validos for p in preguntas if p.tipo == "sin_respuesta")
