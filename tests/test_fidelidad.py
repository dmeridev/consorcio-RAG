"""Tests de la evaluación de fidelidad con un juez falso: no llaman a ninguna API."""

from __future__ import annotations

import json

from rag_consorcio.fidelidad import juzgar, resumir


class JuezFalso:
    def __init__(self, afirmaciones):
        self.afirmaciones = afirmaciones
        self.usuarios = []

    def completar(self, sistema, usuario, esquema):
        self.usuarios.append(usuario)
        return json.dumps({"afirmaciones": self.afirmaciones}), {}


def _respuesta(citados: list[str]) -> dict:
    return {
        "pregunta": "¿Hay carencia?",
        "texto": "No hay carencia si no hay interrupción.",
        "citas": [{"fragmento_id": i} for i in citados],
        "fragmentos": [
            {"id": "a#1", "texto": "TEXTO DEL ART. 8", "unidades": ["a"]},
            {"id": "b#1", "texto": "TEXTO DEL ART. 9", "unidades": ["b"]},
        ],
    }


def test_el_juez_solo_ve_los_fragmentos_citados():
    juez = JuezFalso([{"afirmacion": "x", "respaldada": True, "motivo": ""}])
    juzgar(_respuesta(["a#1"]), juez)
    assert "TEXTO DEL ART. 8" in juez.usuarios[0]
    assert "TEXTO DEL ART. 9" not in juez.usuarios[0]


def test_sin_citas_no_hay_respaldo_y_no_se_llama_al_juez():
    juez = JuezFalso([])
    afirmaciones = juzgar(_respuesta([]), juez)
    assert afirmaciones[0]["respaldada"] is False
    assert juez.usuarios == []


def _caso(id_, con_respuesta, recuperado, respondida, respaldos):
    return {
        "id": id_,
        "con_respuesta": con_respuesta,
        "articulo_recuperado": recuperado,
        "respondida": respondida,
        "error": None,
        "afirmaciones": [{"afirmacion": "", "respaldada": r, "motivo": ""} for r in respaldos],
    }


def test_resumir_metricas():
    casos = [
        _caso("q1", True, True, True, [True, True]),  # fiel
        _caso("q2", True, True, True, [True, False]),  # una afirmación inventada
        _caso("q3", True, False, True, [False]),  # respondió sin el artículo
        _caso("q4", True, False, False, []),  # se abstuvo sin el artículo
        _caso("q5", False, False, False, []),  # sin respuesta: se abstiene bien
        _caso("q6", False, False, True, [False]),  # sin respuesta: se lo inventa
    ]
    m = resumir(casos)["metricas"]
    assert m["fidelidad"] == 3 / 6
    assert m["respuestas_totalmente_fieles"] == 1 / 4
    assert m["tasa_de_respuesta"] == 3 / 4
    assert m["abstencion_correcta"] == 1 / 2
    assert m["responde_sin_el_articulo"] == 1 / 2


def test_calibracion_compara_con_lo_esperado():
    casos = [_caso("q14", True, True, True, [True]), _caso("q11", True, True, True, [False])]
    calibracion = resumir(casos)["calibracion"]
    assert calibracion["q14"] == {"esperado": "fiel", "juez": "fiel"}
    assert calibracion["q11"] == {"esperado": "no fiel", "juez": "no fiel"}
