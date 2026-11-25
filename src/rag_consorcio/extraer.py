"""De PDF del BOE a texto limpio dividido en unidades (preámbulo, artículos, disposiciones).

Pasos, en orden:

1. Sacar el texto de cada página con PyMuPDF, recordando de qué página es cada línea.
2. Quitar las cabeceras y pies que se repiten en todas las páginas
   («BOLETÍN OFICIAL DEL ESTADO», «LEGISLACIÓN CONSOLIDADA», «Página 4»).
3. Saltar el índice: el texto de verdad empieza en la línea «TEXTO CONSOLIDADO».
4. Unir las líneas cortadas a mitad de frase en párrafos.
5. Reconocer los rótulos («Artículo 8. Plazo de carencia.», «Disposición final
   primera...», «CAPÍTULO II ...») y repartir los párrafos en unidades.

El resultado sirve a las dos estrategias de troceado: la de artículos usa las unidades;
la de tamaño fijo usa el texto limpio entero, sin mirar la estructura.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from rag_consorcio.corpus import Documento
from rag_consorcio.numeros import normalizar_numero

_PIES = re.compile(r"^(BOLETÍN OFICIAL DEL ESTADO|LEGISLACIÓN CONSOLIDADA|Página \d+)$")
_INICIO = "TEXTO CONSOLIDADO"

_ARTICULO = re.compile(r"^Artículo (?P<num>[^.]{1,40}?)\.\s*(?P<titulo>.*)$")
_DISPOSICION = re.compile(
    r"^Disposición (?P<clase>adicional|transitoria|derogatoria|final)"
    r"(?: (?P<ord>[a-záéíóú]+))?\.\s*(?P<titulo>.*)$"
)
_ESTRUCTURA = re.compile(r"^(TÍTULO|TITULO|CAPÍTULO|Sección)\s+\S+")
_NIVELES = {"TÍTULO": 0, "TITULO": 0, "CAPÍTULO": 1, "Sección": 2}
# Un número de artículo válido: cifras, «único» o palabras («treinta y tres a)»).
_NUMERO_VALIDO = re.compile(r"^(\d+|único|[a-záéíóúñ ]+)( (bis|ter|quáter|[a-z]\)))?$")
# Lo que abre un párrafo nuevo aunque la línea anterior no termine en punto.
_APARTADO = re.compile(r"^(\d+\.(ª|º)?|[a-zñ]\)|[A-Z]\)|\d+\.\d+\.?)\s")


@dataclass
class Unidad:
    """Un trozo con sentido propio: el preámbulo, un artículo o una disposición."""

    id: str
    tipo: str  # preambulo | articulo | disposicion
    cita: str  # «Reglamento del seguro de riesgos extraordinarios, art. 8»
    titulo: str  # «Plazo de carencia»
    contexto: str  # «CAPÍTULO II Riesgos extraordinarios...» (dónde está dentro de la norma)
    pagina: int
    parrafos: list[str] = field(default_factory=list)
    inicio: int = 0  # posición en `DocumentoLimpio.texto`
    fin: int = 0

    @property
    def texto(self) -> str:
        return "\n".join(self.parrafos)


@dataclass
class DocumentoLimpio:
    documento: Documento
    texto: str
    unidades: list[Unidad]
    paginas: list[tuple[int, int]]
    """(posición en `texto`, página) de cada párrafo, para saber la página de un trozo."""

    def pagina_en(self, posicion: int) -> int:
        pagina = self.paginas[0][1]
        for inicio, pag in self.paginas:
            if inicio > posicion:
                break
            pagina = pag
        return pagina


def _lineas(documento: Documento) -> list[tuple[str, int]]:
    """Líneas del cuerpo (sin índice, cabeceras ni pies), cada una con su página."""
    lineas: list[tuple[str, int]] = []
    with pymupdf.open(documento.ruta) as pdf:
        for numero, pagina in enumerate(pdf, start=1):
            for linea in pagina.get_text().splitlines():
                linea = " ".join(linea.split())
                if linea and not _PIES.match(linea):
                    lineas.append((linea, numero))
    inicio = next(i for i, (linea, _) in enumerate(lineas) if linea == _INICIO)
    cuerpo = lineas[inicio + 1 :]
    if cuerpo and cuerpo[0][0].startswith("Última modificación"):
        cuerpo = cuerpo[1:]
    return cuerpo


def _abre_parrafo(linea: str) -> bool:
    return bool(
        _ARTICULO.match(linea)
        or _DISPOSICION.match(linea)
        or _ESTRUCTURA.match(linea)
        or _APARTADO.match(linea)
    )


def _parrafos(lineas: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Une líneas en párrafos: una línea sigue a la anterior salvo que esta acabe en
    punto, dos puntos o punto y coma, o que la nueva sea un rótulo o un apartado."""
    parrafos: list[tuple[str, int]] = []
    for linea, pagina in lineas:
        if parrafos and not parrafos[-1][0].endswith((".", ":", ";")) and not _abre_parrafo(linea):
            texto, pag = parrafos[-1]
            parrafos[-1] = (f"{texto} {linea}", pag)
        else:
            parrafos.append((linea, pagina))
    return parrafos


def _rotulo(parrafo: str) -> tuple[str, str, str] | None:
    """Si el párrafo es un rótulo de artículo o disposición: (tipo, etiqueta, título)."""
    if m := _ARTICULO.match(parrafo):
        numero = m.group("num").strip()
        if _NUMERO_VALIDO.match(numero.lower()):
            return "articulo", f"art. {normalizar_numero(numero)}", m.group("titulo").strip()
    if m := _DISPOSICION.match(parrafo):
        etiqueta = f"disposición {m.group('clase')}"
        if m.group("ord"):
            etiqueta += f" {m.group('ord')}"
        return "disposicion", etiqueta, m.group("titulo").strip()
    return None


_SIN_CONTENIDO = re.compile(r"^\((Derogad[oa]|Suprimid[oa]|Anulad[oa]|Sin contenido)\)\.?$")


def _sin_contenido(unidad: Unidad) -> bool:
    cuerpo = unidad.parrafos[1:]
    return len(cuerpo) == 1 and bool(_SIN_CONTENIDO.match(cuerpo[0]))


def extraer(documento: Documento) -> DocumentoLimpio:
    parrafos = _parrafos(_lineas(documento))

    # Si el PDF trae delante la norma aprobatoria, empezamos en ella.
    en_texto_aprobado = documento.separador is None
    parte = "texto" if en_texto_aprobado else "norma"
    contexto = ""
    jerarquia: dict[int, str] = {}

    def nombre() -> str:
        return documento.nombre_texto if parte == "texto" else str(documento.nombre_norma)

    unidades: list[Unidad] = [
        Unidad(
            id=f"{documento.id}:{parte}:preambulo",
            tipo="preambulo",
            cita=f"{nombre()}, preámbulo",
            titulo="Preámbulo",
            contexto="",
            pagina=parrafos[0][1],
        )
    ]
    for parrafo, pagina in parrafos:
        # El rótulo del texto aprobado puede venir pegado a la firma del real decreto.
        if not en_texto_aprobado and documento.separador and documento.separador in parrafo:
            parrafo = parrafo[parrafo.index(documento.separador) :]
            en_texto_aprobado, parte, contexto, jerarquia = True, "texto", "", {}
            continue
        if m := _ESTRUCTURA.match(parrafo):
            # Un TÍTULO cierra sus capítulos y secciones; un CAPÍTULO, sus secciones.
            nivel = _NIVELES[m.group(1)]
            jerarquia = {k: v for k, v in jerarquia.items() if k < nivel} | {nivel: parrafo}
            contexto = " › ".join(jerarquia[k] for k in sorted(jerarquia))
            continue
        if rotulo := _rotulo(parrafo):
            tipo, etiqueta, titulo = rotulo
            if tipo == "disposicion":
                # Las disposiciones van al final, fuera de cualquier título o capítulo.
                jerarquia, contexto = {}, ""
            unidades.append(
                Unidad(
                    id=f"{documento.id}:{parte}:{etiqueta}",
                    tipo=tipo,
                    cita=f"{nombre()}, {etiqueta}",
                    titulo=titulo.rstrip(".") or etiqueta,
                    contexto=contexto,
                    pagina=pagina,
                )
            )
        unidades[-1].parrafos.append(parrafo)

    # Fuera las unidades vacías y las derogadas: «Artículo 75. (Derogado)» solo conserva el
    # número, y en una búsqueda haría de cebo para preguntas sobre ese artículo.
    unidades = [u for u in unidades if u.parrafos and not _sin_contenido(u)]
    texto, posiciones = _componer(unidades, parrafos)
    return DocumentoLimpio(documento, texto, unidades, posiciones)


def _componer(
    unidades: list[Unidad], parrafos: list[tuple[str, int]]
) -> tuple[str, list[tuple[int, int]]]:
    """Texto limpio entero, anotando dónde empieza y acaba cada unidad y cada página."""
    pagina_de = {}
    for parrafo, pagina in parrafos:
        pagina_de.setdefault(parrafo, pagina)
    trozos: list[str] = []
    posiciones: list[tuple[int, int]] = []
    cursor = 0
    for unidad in unidades:
        unidad.inicio = cursor
        for parrafo in unidad.parrafos:
            posiciones.append((cursor, pagina_de.get(parrafo, unidad.pagina)))
            trozos.append(parrafo)
            cursor += len(parrafo) + 1
        unidad.fin = cursor
    return "\n".join(trozos), posiciones
