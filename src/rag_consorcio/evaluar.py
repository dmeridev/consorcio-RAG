"""Evaluación de la recuperación: qué combinación trae antes el artículo correcto.

**Conjunto de evaluación** (`data/evaluacion/preguntas.jsonl`): cada pregunta tiene una
lista de `articulos_validos` (puede haber varios: los aerolitos están en el Estatuto y en
el Reglamento) y un `tipo`, para ver si un método gana solo en cierto tipo de pregunta:

- `parafrasis`: dice lo mismo que el artículo con otras palabras.
- `literal`: usa las palabras del artículo. Favorece a BM25.
- `caso`: un caso práctico, como lo contaría un asegurado.
- `referencia`: cita un artículo por su número.
- `sin_respuesta`: el corpus no la responde. No cuenta para la recuperación; servirá para
  comprobar que el LLM sabe decir «no lo sé».

**Métricas** (solo sobre las preguntas con respuesta):

- **Recall@k:** porcentaje de preguntas con algún artículo válido entre los k primeros
  resultados. Recall@5 es la clave: son los fragmentos que recibirá el LLM.
- **MRR** (*Mean Reciprocal Rank*): media de 1/posición del primer acierto (1 si sale
  1.º, 1/2 si sale 2.º…; 0 si no sale entre los 10 primeros). Premia que salga arriba.

Un fragmento acierta si contiene algún artículo válido. Los fragmentos de tamaño fijo
mezclan varios artículos, así que esto les favorece un poco: basta con que contengan un
trozo del artículo correcto, aunque sea una línea.

Uso: `uv run python -m rag_consorcio.evaluar`
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from rag_consorcio.bm25 import Tokenizador
from rag_consorcio.buscar import METODOS, Buscador
from rag_consorcio.corpus import RAIZ
from rag_consorcio.embeddings import E5Local

DIR_EVALUACION = RAIZ / "data" / "evaluacion"
K_MAX = 10
KS = (1, 3, 5)


@dataclass(frozen=True)
class Pregunta:
    id: str
    pregunta: str
    articulos_validos: list[str]
    tipo: str
    origen: str


def cargar_preguntas() -> list[Pregunta]:
    with open(DIR_EVALUACION / "preguntas.jsonl", encoding="utf-8") as f:
        return [Pregunta(**json.loads(linea)) for linea in f]


def posicion_acierto(unidades_por_resultado: list[list[str]], validos: list[str]) -> int | None:
    """Posición (desde 1) del primer resultado que contiene algún artículo válido."""
    for posicion, unidades in enumerate(unidades_por_resultado, start=1):
        if set(unidades) & set(validos):
            return posicion
    return None


def metricas(posiciones: list[int | None]) -> dict[str, float]:
    n = len(posiciones)
    resultado = {f"recall@{k}": sum(1 for p in posiciones if p and p <= k) / n for k in KS}
    resultado["mrr"] = sum(1 / p for p in posiciones if p) / n
    return resultado


def evaluar() -> dict:
    preguntas = [p for p in cargar_preguntas() if p.articulos_validos]
    modelo = E5Local()
    configuraciones = []
    for estrategia in ("articulo", "fijo"):
        for raices in (False, True):
            buscador = Buscador(estrategia, modelo, Tokenizador(raices=raices))
            for metodo in METODOS:
                if metodo == "embeddings" and raices:
                    continue  # las raíces solo afectan a BM25
                configuraciones.append((estrategia, metodo, raices, buscador))

    filas = []
    for estrategia, metodo, raices, buscador in configuraciones:
        posiciones: dict[str, int | None] = {}
        for p in preguntas:
            resultados = buscador.buscar(p.pregunta, metodo, k=K_MAX)
            posiciones[p.id] = posicion_acierto(
                [r.unidades for r in resultados], p.articulos_validos
            )
        por_tipo = defaultdict(list)
        for p in preguntas:
            por_tipo[p.tipo].append(posiciones[p.id])
        filas.append(
            {
                "estrategia": estrategia,
                "metodo": metodo,
                "raices": raices,
                **metricas(list(posiciones.values())),
                "recall@5_por_tipo": {
                    t: metricas(v)["recall@5"] for t, v in sorted(por_tipo.items())
                },
                "posiciones": posiciones,
            }
        )
    return {"fecha": date.today().isoformat(), "preguntas": len(preguntas), "resultados": filas}


def _etiqueta(fila: dict) -> str:
    return f"{fila['estrategia']:8} {fila['metodo']:10} {'raíces' if fila['raices'] else '':6}"


def imprimir(informe: dict) -> None:
    filas = sorted(informe["resultados"], key=lambda f: (-f["recall@5"], -f["mrr"]))
    print(f"\n{informe['preguntas']} preguntas con respuesta\n")
    print(f"{'configuración':28} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    for f in filas:
        print(
            f"{_etiqueta(f):28} {f['recall@1']:6.0%} {f['recall@3']:6.0%} "
            f"{f['recall@5']:6.0%} {f['mrr']:6.2f}"
        )
    tipos = list(filas[0]["recall@5_por_tipo"])
    print(
        f"\nRecall@5 por tipo de pregunta\n{'configuración':28} "
        + " ".join(f"{t:>11}" for t in tipos)
    )
    for f in filas:
        print(f"{_etiqueta(f):28} " + " ".join(f"{f['recall@5_por_tipo'][t]:11.0%}" for t in tipos))


def main() -> None:
    informe = evaluar()
    ruta = DIR_EVALUACION / "resultados.json"
    ruta.write_text(json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8")
    imprimir(informe)
    print(f"\nDetalle por pregunta en {ruta}")


if __name__ == "__main__":
    main()
