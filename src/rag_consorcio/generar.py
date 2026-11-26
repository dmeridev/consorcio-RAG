"""Generación: el LLM redacta la respuesta a partir de los fragmentos recuperados.

Flujo de una pregunta:

1. **Recuperar** los k fragmentos más relevantes (búsqueda híbrida, `buscar.py`).
2. **Montar el prompt:** instrucciones fijas + fragmentos etiquetados [F1], [F2]... con su
   cita + la pregunta.
3. **Llamar al LLM** pidiendo JSON con un esquema estricto:
   `{"respuesta": ..., "citas": ["F1", ...], "respondida": true/false}`.
4. **Verificar en el código** que cada cita es uno de los fragmentos que se le dieron. Una
   instrucción es una petición, no una garantía: si el modelo cita algo que no recibió, se
   lo ha inventado, y eso se descarta y se avisa.

**Proveedor intercambiable.** Groq, OpenAI y Gemini ofrecen APIs compatibles con la de
OpenAI, así que la misma clase sirve para todos cambiando la URL, el modelo y la variable de
entorno con la clave. Por defecto, Groq (plan gratuito, para la demo pública). Para uso
interno con preguntas que puedan contener datos personales habría que usar un plan de pago
que no entrene con los datos.

**Degradación controlada.** Si el LLM falla (sin crédito, límite de peticiones, caída),
`responder()` no lanza la excepción: devuelve los fragmentos encontrados con el error, para
que la demo siga mostrando los artículos aunque no haya respuesta redactada.

Uso: `uv run python -m rag_consorcio.generar "¿Hay carencia si cambio de compañía?"`
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_consorcio.buscar import Buscador, Resultado
from rag_consorcio.corpus import RAIZ

INSTRUCCIONES = """\
Eres un asistente que responde preguntas sobre la normativa del Consorcio de Compensación \
de Seguros (España) y la Ley de Contrato de Seguro.

Reglas:
- Responde SOLO con la información de los fragmentos que se te dan. No uses conocimiento \
propio aunque creas saber la respuesta.
- Cita en `citas` las etiquetas de los fragmentos ([F1], [F2]...) de los que sale la \
respuesta. Solo puedes citar etiquetas que aparezcan en los fragmentos.
- Si los fragmentos no contienen la respuesta, pon `respondida` a false, deja `citas` vacía \
y explica en `respuesta` que la normativa disponible no lo responde. No la inventes.
- Responde en español, de forma clara y breve (3-6 frases), para alguien que no es jurista. \
Menciona el artículo en el texto («según el art. 8 del Reglamento…»).
- Si la respuesta depende de una condición o excepción, dila.
"""

ESQUEMA = {
    "type": "object",
    "properties": {
        "respuesta": {"type": "string"},
        "citas": {"type": "array", "items": {"type": "string"}},
        "respondida": {"type": "boolean"},
    },
    "required": ["respuesta", "citas", "respondida"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Proveedor:
    """Un LLM con API compatible con la de OpenAI."""

    nombre: str
    url: str
    modelo: str
    variable_clave: str


GROQ = Proveedor("groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY")
OPENAI = Proveedor("openai", "https://api.openai.com/v1", "gpt-5.6-luna", "OPENAI_API_KEY")


class ClienteLLM(Protocol):
    def completar(self, sistema: str, usuario: str, esquema: dict) -> tuple[str, dict]:
        """Devuelve el texto JSON de la respuesta y el uso de tokens."""
        ...


@dataclass
class ClienteOpenAICompatible:
    proveedor: Proveedor = GROQ
    timeout: float = 60.0

    def __post_init__(self) -> None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(RAIZ / ".env")
        clave = os.getenv(self.proveedor.variable_clave)
        if not clave:
            raise RuntimeError(
                f"Falta la clave {self.proveedor.variable_clave} en el entorno o en .env"
            )
        self._cliente = OpenAI(
            base_url=self.proveedor.url, api_key=clave, timeout=self.timeout, max_retries=2
        )

    def completar(self, sistema: str, usuario: str, esquema: dict) -> tuple[str, dict]:
        respuesta = self._cliente.chat.completions.create(
            model=self.proveedor.modelo,
            messages=[{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "respuesta_rag", "strict": True, "schema": esquema},
            },
            temperature=0.1,  # poca creatividad: queremos fidelidad a los fragmentos
        )
        uso = respuesta.usage
        return respuesta.choices[0].message.content or "", {
            "entrada": uso.prompt_tokens if uso else 0,
            "salida": uso.completion_tokens if uso else 0,
        }


@dataclass
class Cita:
    etiqueta: str  # «F1»
    cita: str  # «Reglamento del seguro de riesgos extraordinarios, art. 8»
    fragmento_id: str


@dataclass
class Respuesta:
    pregunta: str
    texto: str | None
    """La respuesta redactada; None si el LLM falló."""
    respondida: bool | None
    citas: list[Cita]
    fragmentos: list[Resultado]
    avisos: list[str] = field(default_factory=list)
    error: str | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    segundos: float = 0.0


def etiquetar(fragmentos: list[Resultado]) -> dict[str, Resultado]:
    return {f"F{i}": r for i, r in enumerate(fragmentos, start=1)}


def montar_prompt(pregunta: str, etiquetados: dict[str, Resultado]) -> str:
    bloques = [f"[{etiqueta}]\n{r.fragmento['texto']}" for etiqueta, r in etiquetados.items()]
    return "FRAGMENTOS:\n\n" + "\n\n".join(bloques) + f"\n\nPREGUNTA: {pregunta}"


def verificar(
    datos: dict[str, Any], etiquetados: dict[str, Resultado]
) -> tuple[list[Cita], list[str]]:
    """Se queda solo con las citas que existen de verdad y avisa de lo demás."""
    citas: list[Cita] = []
    avisos: list[str] = []
    for etiqueta in dict.fromkeys(c.strip("[] ") for c in datos.get("citas", [])):
        if etiqueta in etiquetados:
            r = etiquetados[etiqueta]
            citas.append(Cita(etiqueta, r.fragmento["cita"], r.fragmento["id"]))
        else:
            avisos.append(f"El modelo citó «{etiqueta}», que no estaba entre los fragmentos.")
    if datos.get("respondida") and not citas:
        avisos.append("El modelo dice haber respondido pero no cita ningún fragmento válido.")
    if not datos.get("respondida") and citas:
        avisos.append("El modelo dice no haber respondido pero cita fragmentos.")
    return citas, avisos


@dataclass
class Asistente:
    buscador: Buscador = field(default_factory=Buscador)
    cliente: ClienteLLM | None = None
    metodo: str = "rrf"
    k: int = 5

    def __post_init__(self) -> None:
        if self.cliente is None:
            self.cliente = ClienteOpenAICompatible()

    def responder(self, pregunta: str) -> Respuesta:
        fragmentos = self.buscador.buscar(pregunta, self.metodo, self.k)
        etiquetados = etiquetar(fragmentos)
        inicio = time.monotonic()
        try:
            texto, tokens = self.cliente.completar(
                INSTRUCCIONES, montar_prompt(pregunta, etiquetados), ESQUEMA
            )
            datos = json.loads(texto)
        # Cualquier fallo del LLM (red, cuota, JSON roto) degrada a «solo fragmentos».
        except Exception as exc:  # noqa: BLE001
            return Respuesta(
                pregunta,
                None,
                None,
                [],
                fragmentos,
                error=f"No se pudo generar la respuesta ({type(exc).__name__}): {exc}",
                segundos=time.monotonic() - inicio,
            )
        citas, avisos = verificar(datos, etiquetados)
        return Respuesta(
            pregunta,
            datos.get("respuesta", ""),
            bool(datos.get("respondida")),
            citas,
            fragmentos,
            avisos,
            tokens=tokens,
            segundos=time.monotonic() - inicio,
        )


def imprimir(r: Respuesta) -> None:
    print(f"\n❓ {r.pregunta}\n")
    if r.error:
        print(f"⚠️  {r.error}\n\nArtículos encontrados:")
        for f in r.fragmentos:
            print(f"  · {f.fragmento['cita']}")
        return
    print(r.texto)
    print("\nFuentes:" if r.citas else "\n(sin fuentes)")
    for c in r.citas:
        print(f"  [{c.etiqueta}] {c.cita}")
    for aviso in r.avisos:
        print(f"⚠️  {aviso}")
    print(
        f"\n{r.segundos:.1f} s · {r.tokens.get('entrada', 0)} tokens de entrada, "
        f"{r.tokens.get('salida', 0)} de salida"
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Uso: python -m rag_consorcio.generar "pregunta"')
    imprimir(Asistente().responder(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
