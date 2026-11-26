"""Evaluación de la respuesta: ¿lo que dice está respaldado por los artículos que cita?

**LLM como juez (*LLM-as-a-judge*), afirmación por afirmación** (como RAGAS):

1. El asistente responde cada pregunta de la evaluación (`preguntas.jsonl`).
2. Un **juez distinto** del generador (otro modelo: un juez del mismo modelo comparte sus
   puntos ciegos y tiende a ser indulgente con su propio estilo) parte la respuesta en
   afirmaciones y decide si cada una está respaldada por los fragmentos **citados**.

**Métricas:**

- **Fidelidad:** afirmaciones respaldadas / afirmaciones totales, sobre las respuestas que
  dicen haber respondido.
- **Respuestas totalmente fieles:** porcentaje de respuestas sin ninguna afirmación sin
  respaldo.
- **Abstención correcta:** en las preguntas sin respuesta en el corpus, porcentaje en que
  el asistente dice que no lo sabe.
- **Responde sin el artículo:** preguntas en que la búsqueda no trajo ningún artículo
  válido y aun así respondió. Es el riesgo de alucinación más claro.

**Caché:** cada respuesta generada se guarda (`respuestas.jsonl`) con una clave que
depende de la pregunta, los fragmentos, las instrucciones y el modelo. Volver a juzgar no
regenera nada: el plan gratuito de Groq solo permite unos 8.000 tokens por minuto.

**Calibración mínima:** el juez también se equivoca. Se comprueba con dos casos cuya
verdad se conoce (ver `CALIBRACION`).

Uso: `uv run python -m rag_consorcio.fidelidad` (unos 20-30 minutos la primera vez).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date

from rag_consorcio.bm25 import Tokenizador
from rag_consorcio.buscar import Buscador
from rag_consorcio.evaluar import DIR_EVALUACION, cargar_preguntas
from rag_consorcio.generar import (
    GROQ,
    GROQ_JUEZ,
    INSTRUCCIONES,
    VERSION_INSTRUCCIONES,
    Asistente,
    ClienteLLM,
    ClienteOpenAICompatible,
    Respuesta,
)

INSTRUCCIONES_JUEZ = """\
Eres un revisor que comprueba si una respuesta está respaldada por unos textos legales.

1. Divide la RESPUESTA en afirmaciones sobre el contenido de la normativa, de una en una. \
Ignora las frases que solo citan la fuente («según el art. 8…») o hacen de introducción.
2. Para cada afirmación, decide si los TEXTOS la respaldan: `respaldada` es true solo si \
la afirmación se deduce directamente de los textos. Si los textos no dicen eso, o dicen \
otra cosa, es false, aunque la afirmación sea cierta en la realidad.
3. En `motivo`, explica brevemente en qué parte de los textos te basas, o por qué no.

No uses conocimiento propio: juzga solo contra los TEXTOS.
"""

ESQUEMA_JUEZ = {
    "type": "object",
    "properties": {
        "afirmaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "afirmacion": {"type": "string"},
                    "respaldada": {"type": "boolean"},
                    "motivo": {"type": "string"},
                },
                "required": ["afirmacion", "respaldada", "motivo"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["afirmaciones"],
    "additionalProperties": False,
}

# Casos cuya verdad conocemos: si el juez los falla, sus cifras no son fiables.
CALIBRACION = {
    "q14": "fiel",  # carencia al cambiar de compañía: el art. 8 lo dice tal cual
    "q11": "no fiel",  # granizo: no es un riesgo extraordinario (la 1.ª prueba lo afirmó)
}

RUTA_RESPUESTAS = DIR_EVALUACION / "respuestas.jsonl"
RUTA_JUICIOS = DIR_EVALUACION / "juicios.jsonl"


def _cargar_juicios() -> dict[str, list[dict]]:
    if not RUTA_JUICIOS.exists():
        return {}
    with open(RUTA_JUICIOS, encoding="utf-8") as f:
        return {d["clave"]: d["afirmaciones"] for d in map(json.loads, f)}


def _clave(pregunta: str, fragmentos: list[str], modelo: str) -> str:
    base = json.dumps([pregunta, fragmentos, INSTRUCCIONES, modelo], ensure_ascii=False)
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def _cargar_cache() -> dict[str, dict]:
    if not RUTA_RESPUESTAS.exists():
        return {}
    with open(RUTA_RESPUESTAS, encoding="utf-8") as f:
        return {d["clave"]: d for d in map(json.loads, f)}


def _a_dict(r: Respuesta, clave: str, pregunta_id: str) -> dict:
    return {
        "clave": clave,
        "pregunta_id": pregunta_id,
        "pregunta": r.pregunta,
        "texto": r.texto,
        "respondida": r.respondida,
        "citas": [asdict(c) for c in r.citas],
        "fragmentos": [
            {"id": f.fragmento["id"], "texto": f.fragmento["texto"], "unidades": f.unidades}
            for f in r.fragmentos
        ],
        "avisos": r.avisos,
        "error": r.error,
        "tokens": r.tokens,
    }


def juzgar(respuesta: dict, juez: ClienteLLM) -> list[dict]:
    """Afirmaciones de la respuesta con su veredicto, contra los fragmentos citados."""
    citados = {c["fragmento_id"] for c in respuesta["citas"]}
    textos = [f["texto"] for f in respuesta["fragmentos"] if f["id"] in citados]
    if not textos:  # respondió sin citar nada válido: nada la respalda
        return [{"afirmacion": respuesta["texto"], "respaldada": False, "motivo": "Sin citas."}]
    usuario = (
        "TEXTOS:\n\n"
        + "\n\n---\n\n".join(textos)
        + f"\n\nPREGUNTA: {respuesta['pregunta']}\n\nRESPUESTA: {respuesta['texto']}"
    )
    texto, _ = juez.completar(INSTRUCCIONES_JUEZ, usuario, ESQUEMA_JUEZ)
    return json.loads(texto)["afirmaciones"]


def evaluar(asistente: Asistente | None = None, juez: ClienteLLM | None = None) -> dict:
    asistente = asistente or Asistente(
        buscador=Buscador(tokenizador=Tokenizador(raices=True)),
        cliente=ClienteOpenAICompatible(GROQ, reintentos=10),
    )
    # El plan gratuito de Groq limita la salida del juez a 1.000 tokens por minuto: se acota
    # cada juicio para que una sola petición no supere ese límite.
    juez = juez or ClienteOpenAICompatible(
        GROQ_JUEZ, temperatura=0.0, reintentos=10, max_tokens=800
    )
    modelo = getattr(getattr(asistente.cliente, "proveedor", None), "modelo", "desconocido")
    modelo_juez = getattr(getattr(juez, "proveedor", None), "modelo", "desconocido")
    cache = _cargar_cache()
    juicios = _cargar_juicios()

    casos = []
    for p in cargar_preguntas():
        fragmentos = asistente.buscador.buscar(p.pregunta, asistente.metodo, asistente.k)
        clave = _clave(p.pregunta, [f.fragmento["id"] for f in fragmentos], modelo)
        if clave not in cache:
            generada = _a_dict(asistente.responder(p.pregunta), clave, p.id)
            if generada["error"]:  # los fallos no se guardan: se reintentan la próxima vez
                print(f"  {p.id} ERROR al generar: {generada['error'][:120]}")
            else:
                cache[clave] = generada
                with open(RUTA_RESPUESTAS, "a", encoding="utf-8") as f:
                    f.write(json.dumps(generada, ensure_ascii=False) + "\n")
                print(f"  {p.id} generada")
        r = cache.get(clave) or generada
        recuperado = any(set(f["unidades"]) & set(p.articulos_validos) for f in r["fragmentos"])
        caso = {
            "id": p.id,
            "tipo": p.tipo,
            "pregunta": p.pregunta,
            "con_respuesta": bool(p.articulos_validos),
            "articulo_recuperado": recuperado,
            "respondida": r["respondida"],
            "error": r["error"],
            "texto": r["texto"],
            "citas": [c["cita"] for c in r["citas"]],
            "avisos": r["avisos"],
            "afirmaciones": [],
            "error_juez": None,
        }
        if r["respondida"] and not r["error"]:
            clave_juicio = f"{clave}:{modelo_juez}"
            if clave_juicio not in juicios:
                try:
                    juicios[clave_juicio] = juzgar(r, juez)
                except Exception as exc:  # noqa: BLE001 — un fallo no detiene la evaluación
                    caso["error_juez"] = f"{type(exc).__name__}: {exc}"[:300]
                    print(f"  {p.id} ERROR al juzgar: {caso['error_juez'][:120]}")
                else:
                    with open(RUTA_JUICIOS, "a", encoding="utf-8") as f:
                        linea = {"clave": clave_juicio, "afirmaciones": juicios[clave_juicio]}
                        f.write(json.dumps(linea, ensure_ascii=False) + "\n")
                    print(f"  {p.id} juzgada")
            caso["afirmaciones"] = juicios.get(clave_juicio, [])
        casos.append(caso)
    return {
        "fecha": date.today().isoformat(),
        "modelo": modelo,
        "instrucciones": VERSION_INSTRUCCIONES,
        **resumir(casos),
        "casos": casos,
    }


def resumir(casos: list[dict]) -> dict:
    respondidas = [
        c for c in casos if c["respondida"] and not c["error"] and not c.get("error_juez")
    ]
    afirmaciones = [a for c in respondidas for a in c["afirmaciones"]]
    con_respuesta = [c for c in casos if c["con_respuesta"]]
    sin_respuesta = [c for c in casos if not c["con_respuesta"]]
    sin_articulo = [c for c in con_respuesta if not c["articulo_recuperado"]]

    def pct(n: int, d: int) -> float | None:
        return n / d if d else None

    calibracion = {}
    for qid, esperado in CALIBRACION.items():
        caso = next((c for c in casos if c["id"] == qid), None)
        if caso and caso["afirmaciones"]:
            fiel = all(a["respaldada"] for a in caso["afirmaciones"])
            calibracion[qid] = {"esperado": esperado, "juez": "fiel" if fiel else "no fiel"}
    return {
        "metricas": {
            "preguntas": len(casos),
            "errores": sum(1 for c in casos if c["error"]),
            "errores_juez": sum(1 for c in casos if c.get("error_juez")),
            "fidelidad": pct(sum(a["respaldada"] for a in afirmaciones), len(afirmaciones)),
            "respuestas_totalmente_fieles": pct(
                sum(all(a["respaldada"] for a in c["afirmaciones"]) for c in respondidas),
                len(respondidas),
            ),
            "tasa_de_respuesta": pct(
                sum(1 for c in con_respuesta if c["respondida"]), len(con_respuesta)
            ),
            "abstencion_correcta": pct(
                sum(1 for c in sin_respuesta if c["respondida"] is False), len(sin_respuesta)
            ),
            "responde_sin_el_articulo": pct(
                sum(1 for c in sin_articulo if c["respondida"]), len(sin_articulo)
            ),
        },
        "calibracion": calibracion,
    }


def imprimir(informe: dict) -> None:
    m = informe["metricas"]

    def fmt(x):
        return "—" if x is None else f"{x:.0%}"

    print(
        f"\nModelo generador: {informe['modelo']} · instrucciones "
        f"{informe.get('instrucciones', 'v1')} · {m['preguntas']} preguntas"
    )
    print(f"  Fidelidad (afirmaciones respaldadas):     {fmt(m['fidelidad'])}")
    print(f"  Respuestas totalmente fieles:             {fmt(m['respuestas_totalmente_fieles'])}")
    print(f"  Tasa de respuesta (preguntas con resp.):  {fmt(m['tasa_de_respuesta'])}")
    print(f"  Abstención correcta (sin respuesta):      {fmt(m['abstencion_correcta'])}")
    print(f"  Responde sin haber recuperado el artículo: {fmt(m['responde_sin_el_articulo'])}")
    print(f"  Errores del LLM: {m['errores']} · errores del juez: {m['errores_juez']}")
    print("\nCalibración del juez:")
    for qid, c in informe["calibracion"].items():
        ok = "✅" if c["esperado"] == c["juez"] else "❌"
        print(f"  {ok} {qid}: esperado «{c['esperado']}», el juez dice «{c['juez']}»")
    print("\nAfirmaciones sin respaldo:")
    for c in informe["casos"]:
        for a in c["afirmaciones"]:
            if not a["respaldada"]:
                print(f"  [{c['id']}] {a['afirmacion'][:140]}")


def main() -> None:
    informe = evaluar()
    ruta = DIR_EVALUACION / f"fidelidad_{VERSION_INSTRUCCIONES}.json"
    ruta.write_text(json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8")
    imprimir(informe)
    print(f"\nDetalle en {ruta}")


if __name__ == "__main__":
    main()
