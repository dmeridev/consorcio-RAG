"""Embeddings: convertir textos en vectores que representan su significado.

El modelo es **intercambiable**: todo el proyecto habla con `ModeloEmbeddings`, así que
cambiar de modelo local a uno de OpenAI es escribir otra clase con los mismos métodos.

Modelo por defecto: `intfloat/multilingual-e5-small`.

- **Local y gratuito:** se descarga una vez y funciona en CPU. La demo pública no depende
  de ninguna API de pago y los textos no salen del equipo.
- **Multilingüe:** entrenado con muchos idiomas, incluido el español.
- **384 números por vector.** Pequeño y rápido; con 200 fragmentos sobra.
- **Lee como máximo 512 tokens.** Lo que pase de ahí se corta sin avisar; por eso el
  troceado limita los fragmentos a 2.000 caracteres.
- **Prefijos obligatorios:** los modelos E5 se entrenaron viendo `query: ` delante de las
  preguntas y `passage: ` delante de los textos donde se busca. Si no se ponen, el modelo
  funciona peor. Es un detalle de este modelo que hay que leer en su documentación.

Uso: `uv run python -m rag_consorcio.embeddings` (calcula y guarda los vectores).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_consorcio.corpus import DIR_FRAGMENTOS, RAIZ

DIR_INDICES = RAIZ / "data" / "indices"
MODELO_POR_DEFECTO = "intfloat/multilingual-e5-small"


class ModeloEmbeddings(Protocol):
    nombre: str

    def pasajes(self, textos: list[str]) -> np.ndarray:
        """Vectores de los textos donde se busca (los fragmentos)."""
        ...

    def consulta(self, texto: str) -> np.ndarray:
        """Vector de una pregunta."""
        ...


@dataclass
class E5Local:
    """Modelo E5 de sentence-transformers, ejecutado en local."""

    nombre: str = MODELO_POR_DEFECTO

    @cached_property
    def _modelo(self):
        # Se importa aquí para que cargar el resto del proyecto no exija PyTorch.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.nombre)

    def pasajes(self, textos: list[str]) -> np.ndarray:
        return self._codificar([f"passage: {t}" for t in textos])

    def consulta(self, texto: str) -> np.ndarray:
        return self._codificar([f"query: {texto}"])[0]

    def _codificar(self, textos: list[str]) -> np.ndarray:
        # normalize_embeddings=True deja cada vector con longitud 1. Así la similitud del
        # coseno es un simple producto escalar (más rápido y sin dividir).
        return np.asarray(
            self._modelo.encode(textos, normalize_embeddings=True, batch_size=16),
            dtype=np.float32,
        )


def similitud(consulta: np.ndarray, vectores: np.ndarray) -> np.ndarray:
    """Similitud del coseno entre una consulta y cada vector (ya normalizados).

    1 = mismo significado; 0 = nada que ver. Con E5, en la práctica casi todo cae entre
    0,7 y 0,9: lo que cuenta es el orden, no el valor absoluto.
    """
    return vectores @ consulta


def ruta_indice(estrategia: str, modelo: str = MODELO_POR_DEFECTO) -> Path:
    return DIR_INDICES / modelo.replace("/", "__") / f"{estrategia}.npz"


def indexar(estrategia: str, modelo: ModeloEmbeddings) -> Path:
    """Calcula el vector de cada fragmento de una estrategia y lo guarda junto a su id."""
    with open(DIR_FRAGMENTOS / f"{estrategia}.jsonl", encoding="utf-8") as f:
        fragmentos = [json.loads(linea) for linea in f]
    vectores = modelo.pasajes([fr["texto"] for fr in fragmentos])
    ruta = ruta_indice(estrategia, modelo.nombre)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, vectores=vectores, ids=np.array([fr["id"] for fr in fragmentos]))
    return ruta


def cargar_indice(
    estrategia: str, modelo: str = MODELO_POR_DEFECTO
) -> tuple[np.ndarray, list[str]]:
    datos = np.load(ruta_indice(estrategia, modelo))
    return datos["vectores"], datos["ids"].tolist()


def main() -> None:
    modelo = E5Local()
    for estrategia in ("articulo", "fijo"):
        ruta = indexar(estrategia, modelo)
        vectores, _ = cargar_indice(estrategia)
        print(
            f"{estrategia:9} {vectores.shape[0]} vectores de {vectores.shape[1]} números → {ruta}"
        )


if __name__ == "__main__":
    main()
