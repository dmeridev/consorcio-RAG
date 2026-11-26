"""Tests de la generación con un LLM falso: no llaman a ninguna API."""

from __future__ import annotations

import json

from rag_consorcio.buscar import Resultado
from rag_consorcio.generar import Asistente, etiquetar, montar_prompt, verificar


def _resultado(n: int) -> Resultado:
    fragmento = {
        "id": f"reglamento:texto:art. {n}#1",
        "texto": f"Reglamento..., art. {n}\n\nTexto del artículo {n}.",
        "cita": f"Reglamento del seguro de riesgos extraordinarios, art. {n}",
        "unidades": [f"reglamento:texto:art. {n}"],
    }
    return Resultado(fragmento, 1.0, "rrf")


class BuscadorFalso:
    def buscar(self, pregunta, metodo, k):
        return [_resultado(8), _resultado(9)]


class LLMFalso:
    def __init__(self, respuesta):
        self.respuesta = respuesta
        self.prompts = []

    def completar(self, sistema, usuario, esquema):
        self.prompts.append(usuario)
        if isinstance(self.respuesta, Exception):
            raise self.respuesta
        return json.dumps(self.respuesta), {"entrada": 100, "salida": 20}


def _asistente(respuesta) -> tuple[Asistente, LLMFalso]:
    llm = LLMFalso(respuesta)
    return Asistente(buscador=BuscadorFalso(), cliente=llm), llm


def test_el_prompt_lleva_los_fragmentos_etiquetados_y_la_pregunta():
    prompt = montar_prompt("¿Hay carencia?", etiquetar([_resultado(8), _resultado(9)]))
    assert "[F1]\nReglamento..., art. 8" in prompt
    assert "[F2]" in prompt
    assert prompt.endswith("PREGUNTA: ¿Hay carencia?")


def test_cita_valida_se_traduce_al_articulo():
    asistente, _ = _asistente({"respuesta": "No.", "citas": ["F1"], "respondida": True})
    r = asistente.responder("¿Hay carencia si cambio de compañía?")
    assert [c.cita for c in r.citas] == ["Reglamento del seguro de riesgos extraordinarios, art. 8"]
    assert r.avisos == []


def test_cita_inventada_se_descarta_y_se_avisa():
    citas, avisos = verificar(
        {"respuesta": "...", "citas": ["F1", "F7"], "respondida": True},
        etiquetar([_resultado(8), _resultado(9)]),
    )
    assert [c.etiqueta for c in citas] == ["F1"]
    assert any("F7" in a for a in avisos)


def test_acepta_citas_con_corchetes_y_sin_duplicados():
    citas, _ = verificar(
        {"respuesta": "...", "citas": ["[F2]", "F2"], "respondida": True},
        etiquetar([_resultado(8), _resultado(9)]),
    )
    assert [c.etiqueta for c in citas] == ["F2"]


def test_respondida_sin_citas_genera_aviso():
    _, avisos = verificar(
        {"respuesta": "Sí.", "citas": [], "respondida": True}, etiquetar([_resultado(8)])
    )
    assert avisos


def test_sin_respuesta_en_el_corpus():
    asistente, _ = _asistente(
        {"respuesta": "La normativa no lo indica.", "citas": [], "respondida": False}
    )
    r = asistente.responder("¿Qué teléfono tiene el Consorcio?")
    assert r.respondida is False
    assert r.citas == [] and r.avisos == []


def test_si_el_llm_falla_devuelve_los_fragmentos():
    asistente, _ = _asistente(RuntimeError("429 límite de peticiones"))
    r = asistente.responder("¿Hay carencia?")
    assert r.texto is None
    assert "429" in r.error
    assert len(r.fragmentos) == 2


def test_si_el_llm_devuelve_json_roto_tambien_degrada():
    llm = LLMFalso({})
    llm.completar = lambda *a: ("esto no es json", {})
    r = Asistente(buscador=BuscadorFalso(), cliente=llm).responder("¿Hay carencia?")
    assert r.error and r.fragmentos
