"""BM25: búsqueda por palabras exactas, implementada a mano para entender cada pieza.

Los embeddings buscan por significado pero casi ignoran los datos exactos («artículo 9»,
«7 %»). BM25 hace lo contrario: puntúa cada fragmento por las palabras de la pregunta que
contiene, y no entiende sinónimos. Por eso, más adelante, se combinan los dos.

Puntuación de un fragmento D para una pregunta Q (variante de Lucene/Elasticsearch):

    score(D, Q) = Σ  IDF(t) · f(t, D) · (k1 + 1) / ( f(t, D) + k1 · (1 − b + b · |D| / avgdl) )
                 t∈Q

- **f(t, D):** cuántas veces aparece la palabra t en el fragmento. Más veces, más puntos,
  pero con rendimiento decreciente: el término satura (lo controla k1).
- **IDF(t) = ln(1 + (N − n + 0,5) / (n + 0,5)):** N fragmentos en total, n los que
  contienen t. Las palabras raras pesan mucho («franquicia», en 2 de 215: 4,46) y las
  comunes casi nada («seguro», en 185 de 215: 0,15).
- **|D| / avgdl:** longitud del fragmento frente a la media. Un fragmento largo contiene
  muchas palabras por casualidad, así que se le penaliza un poco (lo controla b).
- **k1 = 1,5 y b = 0,75** son los valores habituales.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Palabras tan comunes que no ayudan a distinguir un fragmento de otro. El IDF ya les daría
# casi 0, pero quitarlas hace la búsqueda más limpia y rápida. No se quita «no»: en un
# texto legal cambia el sentido.
PALABRAS_VACIAS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales
    cuando de del desde donde durante e el ella ellas ellos en entre era es esa esas ese
    eso esos esta estas este esto estos fue fueron ha han hasta la las le les lo los mas
    me mi mismo mucho muy nos o os otra otras otro otros para pero poco por porque que
    quien se segun ser si sido sin sobre su sus tambien tanto te tiene tienen todo todos
    tras tu un una uno unos y ya
    """.split()
)

_PALABRA = re.compile(r"\w+")


def _sin_tildes(texto: str) -> str:
    """Quita tildes y diéresis pero conserva la ñ (no es una tilde: es otra letra)."""
    texto = texto.replace("ñ", "\0")
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return texto.replace("\0", "ñ")


@dataclass
class Tokenizador:
    """Convierte un texto en la lista de palabras que compara BM25.

    `raices=True` reduce cada palabra a su raíz con el algoritmo Snowball para español:
    «asegurado», «asegurada» y «asegurados» → «asegur». Ayuda a encontrar variantes, pero a
    veces junta palabras distintas. Se decide con la evaluación.
    """

    raices: bool = False
    _stemmer: object = field(default=None, init=False, repr=False)

    def __call__(self, texto: str) -> list[str]:
        palabras = [
            p
            for p in _PALABRA.findall(_sin_tildes(texto.lower()))
            if _sin_tildes(p) not in PALABRAS_VACIAS
        ]
        if self.raices:
            if self._stemmer is None:
                import snowballstemmer

                self._stemmer = snowballstemmer.stemmer("spanish")
            palabras = self._stemmer.stemWords(palabras)
        return palabras


class BM25:
    def __init__(self, documentos: list[list[str]], k1: float = 1.5, b: float = 0.75):
        """`documentos`: cada fragmento ya tokenizado."""
        self.k1, self.b = k1, b
        self.frecuencias = [Counter(doc) for doc in documentos]
        self.longitudes = np.array([len(doc) for doc in documentos], dtype=float)
        self.longitud_media = float(self.longitudes.mean())
        n_documentos = len(documentos)
        # En cuántos fragmentos aparece cada palabra (una vez por fragmento).
        aparece_en = Counter(palabra for doc in documentos for palabra in set(doc))
        self.idf = {
            palabra: math.log(1 + (n_documentos - n + 0.5) / (n + 0.5))
            for palabra, n in aparece_en.items()
        }

    def puntuar(self, consulta: list[str]) -> np.ndarray:
        """Puntuación BM25 de cada fragmento para la consulta (0 si no comparte palabras)."""
        puntuaciones = np.zeros(len(self.frecuencias))
        normalizacion = self.k1 * (1 - self.b + self.b * self.longitudes / self.longitud_media)
        for palabra in set(consulta):
            idf = self.idf.get(palabra)
            if idf is None:  # la palabra no sale en ningún fragmento
                continue
            f = np.array([frec[palabra] for frec in self.frecuencias], dtype=float)
            puntuaciones += idf * f * (self.k1 + 1) / (f + normalizacion)
        return puntuaciones
