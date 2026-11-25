"""Recuperación: dada una pregunta, los k fragmentos más relevantes.

Cuatro métodos, para poder compararlos en la evaluación:

- `embeddings`: por significado (similitud del coseno).
- `bm25`: por palabras exactas.
- `rrf` (el principal): **Reciprocal Rank Fusion**. Combina las dos listas mirando solo la
  posición de cada fragmento en cada una:

      RRF(d) = Σ  1 / (60 + posición de d en la lista)
              listas

  Premia a los fragmentos que salen arriba en las dos. No usa las puntuaciones, así que no
  importa que una vaya de 0,8 a 0,9 y la otra de 0 a 15, y no hay nada que ajustar. El 60
  es el valor estándar: suaviza la diferencia entre el 1.º y el 2.º puesto. Es el método
  de búsqueda híbrida de Elasticsearch, OpenSearch y la mayoría de bases vectoriales.

- `ponderada` (para comparar): normaliza cada lista de 0 a 1 (min-max) y suma
  `alfa · embeddings + (1 − alfa) · BM25`. Usa las puntuaciones, pero hay que elegir alfa y
  un valor extremo puede aplastar al resto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from rag_consorcio.bm25 import BM25, Tokenizador
from rag_consorcio.corpus import DIR_FRAGMENTOS
from rag_consorcio.embeddings import E5Local, ModeloEmbeddings, cargar_indice, similitud

METODOS = ("embeddings", "bm25", "rrf", "ponderada")
K_RRF = 60


def fusionar_rrf(rankings: list[list[int]], k: int = K_RRF) -> dict[int, float]:
    """Puntuación RRF de cada fragmento a partir de varias listas ordenadas de índices.

    Un fragmento que no aparece en una lista no suma nada por ella.
    """
    puntuaciones: dict[int, float] = {}
    for ranking in rankings:
        for posicion, indice in enumerate(ranking, start=1):
            puntuaciones[indice] = puntuaciones.get(indice, 0.0) + 1 / (k + posicion)
    return puntuaciones


def normalizar_min_max(puntuaciones: np.ndarray) -> np.ndarray:
    """Reescala a [0, 1]. Si todas son iguales (p. ej., todo ceros en BM25), devuelve ceros."""
    minimo, maximo = puntuaciones.min(), puntuaciones.max()
    if maximo == minimo:
        return np.zeros_like(puntuaciones)
    return (puntuaciones - minimo) / (maximo - minimo)


@dataclass
class Resultado:
    fragmento: dict
    puntuacion: float
    metodo: str

    @property
    def unidades(self) -> list[str]:
        return self.fragmento["unidades"]


@dataclass
class Buscador:
    """Busca sobre los fragmentos de una estrategia de troceado («articulo» o «fijo»)."""

    estrategia: str = "articulo"
    modelo: ModeloEmbeddings = field(default_factory=E5Local)
    tokenizador: Tokenizador = field(default_factory=Tokenizador)
    alfa: float = 0.5

    def __post_init__(self) -> None:
        with open(DIR_FRAGMENTOS / f"{self.estrategia}.jsonl", encoding="utf-8") as f:
            self.fragmentos = [json.loads(linea) for linea in f]
        self.vectores, ids = cargar_indice(self.estrategia, self.modelo.nombre)
        if ids != [fr["id"] for fr in self.fragmentos]:
            raise RuntimeError(
                "El índice de embeddings no corresponde a los fragmentos actuales: "
                "vuelve a ejecutar `python -m rag_consorcio.embeddings`."
            )

    @cached_property
    def _bm25(self) -> BM25:
        return BM25([self.tokenizador(fr["texto"]) for fr in self.fragmentos])

    def puntuaciones_embeddings(self, pregunta: str) -> np.ndarray:
        return similitud(self.modelo.consulta(pregunta), self.vectores)

    def puntuaciones_bm25(self, pregunta: str) -> np.ndarray:
        return self._bm25.puntuar(self.tokenizador(pregunta))

    def buscar(self, pregunta: str, metodo: str = "rrf", k: int = 5) -> list[Resultado]:
        if metodo not in METODOS:
            raise ValueError(f"Método desconocido: {metodo}. Opciones: {METODOS}")

        if metodo == "embeddings":
            puntuaciones = self.puntuaciones_embeddings(pregunta)
        elif metodo == "bm25":
            puntuaciones = self.puntuaciones_bm25(pregunta)
        elif metodo == "ponderada":
            puntuaciones = self.alfa * normalizar_min_max(
                self.puntuaciones_embeddings(pregunta)
            ) + (1 - self.alfa) * normalizar_min_max(self.puntuaciones_bm25(pregunta))
        else:  # rrf
            emb = self.puntuaciones_embeddings(pregunta)
            bm25 = self.puntuaciones_bm25(pregunta)
            ranking_emb = np.argsort(-emb, kind="stable").tolist()
            # En BM25 solo cuentan los fragmentos que comparten alguna palabra con la
            # pregunta: los de puntuación 0 no están «en la lista».
            ranking_bm25 = [i for i in np.argsort(-bm25, kind="stable") if bm25[i] > 0]
            fusion = fusionar_rrf([ranking_emb, ranking_bm25])
            puntuaciones = np.array([fusion.get(i, 0.0) for i in range(len(self.fragmentos))])

        mejores = np.argsort(-puntuaciones, kind="stable")[:k]
        return [Resultado(self.fragmentos[i], float(puntuaciones[i]), metodo) for i in mejores]
