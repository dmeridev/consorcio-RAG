"""Tests de la recuperación: la fusión RRF, la normalización y la búsqueda real."""

from __future__ import annotations

import numpy as np
import pytest

from rag_consorcio.buscar import fusionar_rrf, normalizar_min_max


def test_rrf_premia_estar_arriba_en_las_dos_listas():
    # El 7 es 1.º solo en la primera lista; el 3 es 3.º en las dos.
    puntuaciones = fusionar_rrf([[7, 1, 3], [2, 1, 3]])
    assert puntuaciones[3] == pytest.approx(2 / 63)
    assert puntuaciones[7] == pytest.approx(1 / 61)
    assert puntuaciones[3] > puntuaciones[7]


def test_rrf_quien_no_aparece_en_una_lista_no_suma_por_ella():
    puntuaciones = fusionar_rrf([[0, 1], [1]])
    assert puntuaciones[0] == pytest.approx(1 / 61)
    assert puntuaciones[1] == pytest.approx(1 / 62 + 1 / 61)


def test_normalizar_min_max():
    assert normalizar_min_max(np.array([2.0, 4.0, 6.0])).tolist() == [0.0, 0.5, 1.0]


def test_normalizar_todo_igual_da_ceros():
    assert normalizar_min_max(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def buscador():
    from rag_consorcio.buscar import Buscador

    return Buscador()


@pytest.mark.modelo
@pytest.mark.parametrize("metodo", ["embeddings", "bm25", "rrf", "ponderada"])
def test_todos_los_metodos_encuentran_la_franquicia(buscador, metodo):
    resultados = buscador.buscar("¿Qué franquicia se aplica a los daños en las cosas?", metodo)
    assert len(resultados) == 5
    assert "reglamento:texto:art. 9" in resultados[0].unidades


@pytest.mark.modelo
def test_metodo_desconocido(buscador):
    with pytest.raises(ValueError):
        buscador.buscar("hola", "magia")
