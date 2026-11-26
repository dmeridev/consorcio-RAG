"""Tests de la demo con un asistente falso: no cargan modelos ni llaman a ninguna API."""

from __future__ import annotations

import json

from rag_consorcio.buscar import Resultado
from rag_consorcio.demo import Demo, formatear_fragmentos, formatear_respuesta
from rag_consorcio.generar import VERSION_INSTRUCCIONES, Cita, Respuesta


def _fragmento(n: int) -> Resultado:
    return Resultado(
        {
            "id": f"r:art. {n}#1",
            "texto": f"Reglamento, art. {n}\n\nTexto del artículo {n}.",
            "cita": f"Reglamento, art. {n}",
            "unidades": [f"r:art. {n}"],
        },
        1.0,
        "rrf",
    )


def _respuesta(**cambios) -> Respuesta:
    base = {
        "pregunta": "¿Hay carencia?",
        "texto": "No hay carencia.",
        "respondida": True,
        "citas": [Cita("F1", "Reglamento, art. 8", "r:art. 8#1")],
        "fragmentos": [_fragmento(8), _fragmento(9)],
    }
    return Respuesta(**{**base, **cambios})


class AsistenteFalso:
    def __init__(self, respuesta: Respuesta):
        self.respuesta = respuesta
        self.llamadas = 0

    def responder(self, pregunta):
        self.llamadas += 1
        return self.respuesta


def _demo(respuesta: Respuesta, tmp_path) -> tuple[Demo, AsistenteFalso]:
    asistente = AsistenteFalso(respuesta)
    demo = Demo(asistente)
    demo._cache.clear()  # sin los ejemplos reales
    return demo, asistente


def test_respuesta_con_fuentes():
    texto = formatear_respuesta(_respuesta())
    assert "No hay carencia." in texto
    assert "- Reglamento, art. 8" in texto


def test_abstencion_se_muestra_claramente():
    texto = formatear_respuesta(_respuesta(respondida=False, citas=[], texto="No lo dice."))
    assert "no lo responde de forma expresa" in texto


def test_error_del_llm_muestra_aviso():
    assert "no se puede redactar" in formatear_respuesta(_respuesta(texto=None, error="429"))


def test_fragmentos_marcan_el_citado():
    html = formatear_fragmentos(_respuesta())
    assert "📌 <b>[F1] Reglamento, art. 8</b>" in html
    assert "📌 <b>[F2]" not in html
    assert "Texto del artículo 9." in html


def test_cache_evita_repetir_la_llamada(tmp_path):
    demo, asistente = _demo(_respuesta(), tmp_path)
    demo.responder("¿Hay carencia?")
    demo.responder("  ¿HAY   carencia? ")
    assert asistente.llamadas == 1


def test_los_errores_no_se_guardan_en_cache(tmp_path):
    demo, asistente = _demo(_respuesta(texto=None, error="429"), tmp_path)
    demo.responder("¿Hay carencia?")
    demo.responder("¿Hay carencia?")
    assert asistente.llamadas == 2


def test_validacion_de_la_pregunta(tmp_path):
    demo, asistente = _demo(_respuesta(), tmp_path)
    assert "Escribe" in demo.responder("   ")[0]
    assert "demasiado larga" in demo.responder("x" * 301)[0]
    assert asistente.llamadas == 0


def test_ejemplos_solo_se_cargan_si_coincide_la_version(tmp_path):
    demo, _ = _demo(_respuesta(), tmp_path)
    ruta = tmp_path / "ejemplos.json"
    ejemplo = {"pregunta": "P", "respuesta": "R", "fragmentos": "F"}
    ruta.write_text(json.dumps({"instrucciones": "v0", "ejemplos": [ejemplo]}), encoding="utf-8")
    demo.cargar_ejemplos(ruta)
    assert "p" not in demo._cache
    datos = {"instrucciones": VERSION_INSTRUCCIONES, "ejemplos": [ejemplo]}
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    demo.cargar_ejemplos(ruta)
    assert demo._cache["p"] == ("R", "F")
