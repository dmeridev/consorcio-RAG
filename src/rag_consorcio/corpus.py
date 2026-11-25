"""Los documentos del corpus: qué fichero es cada uno y cómo se cita.

Cada PDF del BOE de una norma que aprueba un texto (un reglamento, un texto refundido)
tiene dos partes:

- la **norma aprobatoria**: el real decreto en sí, con su «Artículo único» y sus
  disposiciones;
- el **texto aprobado**: el reglamento o el estatuto, con sus artículos 1, 2, 3...

Las dos tienen artículos y disposiciones con el mismo nombre (las dos pueden tener una
«Disposición adicional única»), así que hay que saber en qué parte está cada uno para
citarlo bien. `separador` es el rótulo en mayúsculas con el que empieza el texto aprobado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"
DIR_FRAGMENTOS = RAIZ / "data" / "fragmentos"


@dataclass(frozen=True)
class Documento:
    id: str
    fichero: str
    nombre_texto: str
    """Cómo se cita el texto principal (los artículos que importan)."""
    nombre_norma: str | None = None
    """Cómo se cita la norma aprobatoria, si el PDF la incluye delante."""
    separador: str | None = None
    """Rótulo en mayúsculas con el que empieza el texto aprobado."""

    @property
    def ruta(self) -> Path:
        return DIR_RAW / self.fichero


CORPUS: tuple[Documento, ...] = (
    Documento(
        id="estatuto",
        fichero="BOE-A-2004-18910-consolidado.pdf",
        nombre_texto="Estatuto Legal del Consorcio de Compensación de Seguros",
        nombre_norma="Real Decreto Legislativo 7/2004",
        separador="TEXTO REFUNDIDO DEL ESTATUTO LEGAL",
    ),
    Documento(
        id="reglamento",
        fichero="BOE-A-2004-3373-consolidado.pdf",
        nombre_texto="Reglamento del seguro de riesgos extraordinarios",
        nombre_norma="Real Decreto 300/2004",
        separador="REGLAMENTO DEL SEGURO DE RIESGOS EXTRAORDINARIOS",
    ),
    Documento(
        id="lcs",
        fichero="BOE-A-1980-22501-consolidado.pdf",
        nombre_texto="Ley 50/1980 de Contrato de Seguro",
    ),
)
