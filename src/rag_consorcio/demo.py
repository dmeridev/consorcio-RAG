"""Demo web con Gradio: pregunta → respuesta con sus fuentes y los artículos encontrados.

Pensada para publicarse gratis en Hugging Face Spaces, con una cuota de LLM limitada
(unos 200.000 tokens al día; cada respuesta gasta unos 2.000). Por eso:

- **Modelo ligero:** `gpt-oss-20b`, con su propia cuota, en lugar del de 120b que se usa en
  las evaluaciones.
- **Caché de respuestas:** una pregunta repetida no vuelve a llamar al LLM.
- **Ejemplos precalculados** (`data/demo/ejemplos.json`): la mayoría de la gente pulsa los
  ejemplos, y esos no gastan cuota.
- **Degradación controlada:** si el LLM falla o se agota la cuota, se muestran los
  artículos encontrados con un aviso. La búsqueda es local y siempre funciona.

Uso local: `uv run python app.py` y abrir http://localhost:7860
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from rag_consorcio.corpus import DIR_FRAGMENTOS, RAIZ
from rag_consorcio.embeddings import ruta_indice
from rag_consorcio.generar import (
    GROQ,
    VERSION_INSTRUCCIONES,
    Asistente,
    ClienteOpenAICompatible,
    Proveedor,
    Respuesta,
)

GROQ_DEMO = Proveedor(GROQ.nombre, GROQ.url, "openai/gpt-oss-20b", GROQ.variable_clave)
DIR_DEMO = RAIZ / "data" / "demo"
RUTA_EJEMPLOS = DIR_DEMO / "ejemplos.json"
MAX_CARACTERES_PREGUNTA = 300

PREGUNTAS_EJEMPLO = [
    "¿Hay carencia si cambio de compañía?",
    "¿Cubre el Consorcio los daños por terrorismo?",
    "¿Cuántos días tengo para avisar a la aseguradora de un siniestro?",
    "¿Se aplica el plazo de carencia en los seguros de personas?",
    "¿Qué es un tornado a efectos de la cobertura de riesgos extraordinarios?",
    "¿Qué teléfono tiene el Consorcio para declarar un siniestro?",
]

AVISO_LEGAL = (
    "⚠️ **Demo con fines educativos.** Las respuestas las genera una IA a partir de la "
    "normativa y pueden contener errores. No sustituyen el asesoramiento profesional: "
    "comprueba siempre el artículo citado."
)


def preparar_indices() -> None:
    """Genera los fragmentos y los vectores si no existen (primer arranque del Space)."""
    from rag_consorcio import embeddings, trocear

    if not (DIR_FRAGMENTOS / "articulo.jsonl").exists():
        trocear.guardar(trocear.generar())
    modelo = embeddings.modelo_configurado()
    if not ruta_indice("articulo", modelo.nombre).exists():
        embeddings.indexar("articulo", modelo)


def clave(pregunta: str) -> str:
    return " ".join(pregunta.lower().split())


def formatear_respuesta(r: Respuesta) -> str:
    if r.error:
        return (
            "⚠️ **Ahora mismo no se puede redactar la respuesta** (puede que se haya agotado "
            "la cuota gratuita del modelo de lenguaje). Abajo tienes los artículos más "
            "relevantes que ha encontrado la búsqueda."
        )
    if not r.respondida:
        return f"🤷 **La normativa disponible no lo responde de forma expresa.**\n\n{r.texto}"
    partes = [r.texto, "", "**Fuentes:**"]
    partes += [f"- {c.cita}" for c in r.citas]
    if r.avisos:
        partes += ["", *[f"⚠️ {a}" for a in r.avisos]]
    return "\n".join(partes)


def formatear_fragmentos(r: Respuesta) -> str:
    citados = {c.fragmento_id for c in r.citas}
    bloques = []
    for i, f in enumerate(r.fragmentos, start=1):
        marca = "📌 " if f.fragmento["id"] in citados else ""
        cuerpo = f.fragmento["texto"].split("\n\n", 1)[-1]
        bloques.append(
            f"<details><summary>{marca}<b>[F{i}] {f.fragmento['cita']}</b></summary>\n\n"
            f"{cuerpo}\n\n</details>"
        )
    return "\n\n".join(bloques)


class Demo:
    def __init__(self, asistente: Asistente | None = None) -> None:
        self.asistente = asistente or Asistente(
            cliente=ClienteOpenAICompatible(GROQ_DEMO, reintentos=1, timeout=30)
        )
        self._cache: dict[str, tuple[str, str]] = {}
        self._cerrojo = threading.Lock()
        self.cargar_ejemplos()

    def calentar(self) -> None:
        """Carga el modelo de embeddings y construye BM25 al arrancar, no en la primera
        pregunta: si no, el primer usuario espera unos 20 segundos (arranque en frío)."""
        self.asistente.buscador.buscar("calentamiento", self.asistente.metodo, 1)

    def cargar_ejemplos(self, ruta: Path = RUTA_EJEMPLOS) -> None:
        if ruta.exists():
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            if datos.get("instrucciones") == VERSION_INSTRUCCIONES:
                for e in datos["ejemplos"]:
                    self._cache[clave(e["pregunta"])] = (e["respuesta"], e["fragmentos"])

    def responder(self, pregunta: str) -> tuple[str, str]:
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return "Escribe una pregunta sobre la normativa del Consorcio.", ""
        if len(pregunta) > MAX_CARACTERES_PREGUNTA:
            return f"La pregunta es demasiado larga (máximo {MAX_CARACTERES_PREGUNTA}).", ""
        k = clave(pregunta)
        with self._cerrojo:
            if k in self._cache:
                return self._cache[k]
        r = self.asistente.responder(pregunta)
        salida = (formatear_respuesta(r), formatear_fragmentos(r))
        if not r.error:  # los fallos no se guardan: se reintenta en la próxima pregunta
            with self._cerrojo:
                self._cache[k] = salida
        return salida


def generar_ejemplos() -> None:
    """Precalcula las respuestas de los ejemplos (se ejecuta una vez, en local)."""
    # Es un proceso por lotes: aquí sí compensa esperar a que se libere la cuota por minuto.
    demo = Demo(Asistente(cliente=ClienteOpenAICompatible(GROQ_DEMO, reintentos=10)))
    demo._cache.clear()  # regenerar todos, no reutilizar los guardados
    ejemplos = []
    for p in PREGUNTAS_EJEMPLO:
        respuesta, fragmentos = demo.responder(p)
        if respuesta.startswith("⚠️ **Ahora mismo"):
            raise RuntimeError(f"No se pudo generar el ejemplo: {p}")
        ejemplos.append({"pregunta": p, "respuesta": respuesta, "fragmentos": fragmentos})
        print(f"✓ {p}")
    DIR_DEMO.mkdir(parents=True, exist_ok=True)
    RUTA_EJEMPLOS.write_text(
        json.dumps(
            {
                "instrucciones": VERSION_INSTRUCCIONES,
                "modelo": GROQ_DEMO.modelo,
                "ejemplos": ejemplos,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def construir_interfaz(demo: Demo | None = None):
    import gradio as gr

    demo = demo or Demo()
    with gr.Blocks(title="Consorcio RAG") as interfaz:
        gr.Markdown(
            "# Consulta la normativa del Consorcio de Compensación de Seguros\n"
            "Pregunta en lenguaje normal. La respuesta sale del **Estatuto Legal del "
            "Consorcio**, el **Reglamento del seguro de riesgos extraordinarios** y la **Ley "
            "de Contrato de Seguro** (textos consolidados del BOE), y cita el artículo."
        )
        gr.Markdown(AVISO_LEGAL)
        pregunta = gr.Textbox(
            label="Tu pregunta",
            placeholder="Por ejemplo: ¿cubre el Consorcio los daños de una inundación?",
            max_length=MAX_CARACTERES_PREGUNTA,
            lines=2,
        )
        boton = gr.Button("Preguntar", variant="primary")
        respuesta = gr.Markdown(label="Respuesta")
        with gr.Accordion("Artículos encontrados por la búsqueda", open=False):
            fragmentos = gr.Markdown()
        gr.Examples(PREGUNTAS_EJEMPLO, inputs=pregunta, label="Prueba con un ejemplo")
        gr.Markdown(
            "<small>Búsqueda híbrida (embeddings locales + BM25, combinados con Reciprocal "
            "Rank Fusion) y respuesta generada con `gpt-oss-20b` en Groq. Código en "
            "GitHub.</small>"
        )
        boton.click(demo.responder, inputs=pregunta, outputs=[respuesta, fragmentos])
        pregunta.submit(demo.responder, inputs=pregunta, outputs=[respuesta, fragmentos])
    return interfaz


if __name__ == "__main__":
    generar_ejemplos()
