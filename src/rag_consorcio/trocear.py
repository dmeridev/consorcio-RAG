"""Las dos estrategias de troceado (chunking) que vamos a comparar.

**Por artículo** (la principal): un fragmento por artículo o disposición. Los largos se
parten por apartados, y **cada fragmento empieza con una cabecera** que dice de qué norma
y de qué artículo es («Reglamento del seguro de riesgos extraordinarios, art. 8. Plazo de
carencia»). Así ningún fragmento pierde su contexto, aunque sea la tercera parte de un
artículo largo.

**Tamaño fijo** (la de referencia): trozos de `TAMANO` caracteres sobre el texto limpio,
con `SOLAPE` caracteres repetidos entre uno y el siguiente, sin mirar dónde empieza o
acaba cada artículo y sin cabecera. Es lo que se haría sin conocer la estructura.

Las dos guardan en `unidades` qué artículos toca cada fragmento: es lo que usaremos en la
evaluación para saber si la búsqueda ha traído el artículo correcto.

Uso: `uv run python -m rag_consorcio.trocear`
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from rag_consorcio.corpus import CORPUS, DIR_FRAGMENTOS
from rag_consorcio.extraer import DocumentoLimpio, Unidad, extraer

# El modelo de embeddings (multilingual-e5) solo lee 512 tokens: lo que pase de ahí se corta
# sin avisar y no cuenta en la búsqueda. Con 2.000 caracteres el fragmento más largo ronda
# los 470 tokens, con margen. (Con 3.000, 17 fragmentos se pasaban.)
MAX_CARACTERES = 2000
TAMANO = 1000
SOLAPE = 200

_APARTADO_NUMERADO = re.compile(r"^\d+\.\s")


@dataclass
class Fragmento:
    id: str
    estrategia: str  # articulo | fijo
    documento: str
    texto: str
    cita: str
    """Para la respuesta: «Ley 50/1980 de Contrato de Seguro, art. 33 a)»."""
    unidades: list[str]
    """Ids de los artículos o disposiciones que contiene (para la evaluación)."""
    pagina: int
    inicio: int = 0
    """Posición en el texto limpio del documento donde empieza (para depurar y probar)."""
    fin: int = 0


def cabecera(unidad: Unidad, parte: str = "") -> str:
    """«Norma, art. N. Título (parte)» y, debajo, dónde está dentro de la norma."""
    linea = (
        unidad.cita
        if unidad.titulo == unidad.cita.split(", ")[-1]
        else (f"{unidad.cita}. {unidad.titulo}")
    )
    return f"{linea}{parte}\n{unidad.contexto}".rstrip()


def _grupos(parrafos: list[str], maximo: int) -> list[list[str]]:
    """Reparte los párrafos en grupos de hasta `maximo` caracteres.

    Se prefiere cortar donde empieza un apartado numerado («2. ...»); si un apartado no
    cabe solo, se corta entre sus párrafos. Nunca se parte un párrafo por la mitad.
    """
    # Primero, bloques que empiezan en cada apartado numerado.
    bloques: list[list[str]] = []
    for parrafo in parrafos:
        if not bloques or _APARTADO_NUMERADO.match(parrafo):
            bloques.append([parrafo])
        else:
            bloques[-1].append(parrafo)

    grupos: list[list[str]] = [[]]
    largo = 0
    for bloque in bloques:
        piezas = [bloque] if sum(len(p) + 1 for p in bloque) <= maximo else [[p] for p in bloque]
        for pieza in piezas:
            tam = sum(len(p) + 1 for p in pieza)
            if grupos[-1] and largo + tam > maximo:
                grupos.append([])
                largo = 0
            grupos[-1].extend(pieza)
            largo += tam
    return grupos


def por_articulo(doc: DocumentoLimpio) -> list[Fragmento]:
    fragmentos: list[Fragmento] = []
    for unidad in doc.unidades:
        # El primer párrafo es el propio rótulo («Artículo 8. Plazo de carencia.»), que ya
        # va en la cabecera; el preámbulo no tiene rótulo.
        cuerpo = unidad.parrafos if unidad.tipo == "preambulo" else unidad.parrafos[1:]
        hueco = MAX_CARACTERES - len(cabecera(unidad, " (10/10)"))
        grupos = _grupos(cuerpo, hueco) if cuerpo else [[]]
        for n, grupo in enumerate(grupos, start=1):
            parte = f" ({n}/{len(grupos)})" if len(grupos) > 1 else ""
            fragmentos.append(
                Fragmento(
                    id=f"{unidad.id}#{n}",
                    estrategia="articulo",
                    documento=doc.documento.id,
                    texto=f"{cabecera(unidad, parte)}\n\n" + "\n".join(grupo),
                    cita=unidad.cita,
                    unidades=[unidad.id],
                    pagina=unidad.pagina,
                    inicio=unidad.inicio,
                    fin=unidad.fin,
                )
            )
    return fragmentos


def fijo(doc: DocumentoLimpio, tamano: int = TAMANO, solape: int = SOLAPE) -> list[Fragmento]:
    texto = doc.texto
    fragmentos: list[Fragmento] = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            # Cortar en el último espacio para no partir una palabra.
            corte = texto.rfind(" ", inicio + tamano // 2, fin)
            fin = corte if corte != -1 else fin
        tocadas = [u for u in doc.unidades if u.inicio < fin and u.fin > inicio]
        fragmentos.append(
            Fragmento(
                id=f"{doc.documento.id}:fijo#{len(fragmentos) + 1}",
                estrategia="fijo",
                documento=doc.documento.id,
                texto=texto[inicio:fin].strip(),
                cita=f"{doc.documento.nombre_texto} (pág. {doc.pagina_en(inicio)})",
                unidades=[u.id for u in tocadas],
                pagina=doc.pagina_en(inicio),
                inicio=inicio,
                fin=fin,
            )
        )
        if fin >= len(texto):
            break
        inicio = max(fin - solape, inicio + 1)
    return fragmentos


def generar() -> dict[str, list[Fragmento]]:
    resultado: dict[str, list[Fragmento]] = {"articulo": [], "fijo": []}
    for documento in CORPUS:
        doc = extraer(documento)
        resultado["articulo"].extend(por_articulo(doc))
        resultado["fijo"].extend(fijo(doc))
    return resultado


def guardar(resultado: dict[str, list[Fragmento]]) -> None:
    DIR_FRAGMENTOS.mkdir(parents=True, exist_ok=True)
    for estrategia, fragmentos in resultado.items():
        with open(DIR_FRAGMENTOS / f"{estrategia}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(asdict(fragmento), ensure_ascii=False) + "\n" for fragmento in fragmentos
            )


def main() -> None:
    resultado = generar()
    guardar(resultado)
    for estrategia, fragmentos in resultado.items():
        largos = sorted(len(f.texto) for f in fragmentos)
        varios = sum(1 for f in fragmentos if len(f.unidades) > 1)
        print(
            f"{estrategia:9} {len(fragmentos):4} fragmentos · mediana {largos[len(largos) // 2]} "
            f"caracteres · máximo {largos[-1]} · tocan más de un artículo: {varios}"
        )
    print(f"Guardados en {DIR_FRAGMENTOS}")


if __name__ == "__main__":
    main()
