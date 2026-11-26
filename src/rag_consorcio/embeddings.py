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
class E5Onnx:
    """El mismo modelo E5 ejecutado con ONNX Runtime, sin PyTorch.

    PyTorch ocupa casi 1 GB de memoria; ONNX Runtime, una fracción. Eso permite publicar la
    demo en un hosting gratuito. Los pesos son los oficiales del modelo exportados a ONNX
    (`onnx/model.onnx` en su repositorio).

    Lo que sentence-transformers hacía por dentro, aquí se hace a mano:

    1. **Tokenizar** (librería `tokenizers`), cortando en 512 tokens.
    2. **Ejecutar el modelo:** devuelve un vector de 384 números **por cada token**.
    3. **Mean pooling:** la media de los vectores de los tokens, sin contar el relleno
       (*padding*) que se añade para que todos los textos del lote midan lo mismo. La
       máscara de atención dice qué tokens son reales.
    4. **Normalizar** a longitud 1.

    Por defecto, los pesos en precisión completa (fp32, ~450 MB en memoria). `E5Onnx.int8()`
    usa una versión **cuantizada a 8 bits** (~115 MB), con vectores ligeramente distintos:
    hay que evaluarla antes de usarla.
    """

    nombre: str = MODELO_POR_DEFECTO
    archivo_modelo: str = "onnx/model.onnx"
    archivo_tokenizador: str = "onnx/tokenizer.json"
    lote: int = 16

    @classmethod
    def int8(cls) -> E5Onnx:
        """Cuantización dinámica a int8 del mismo modelo (exportada por Xenova)."""
        return cls("Xenova/multilingual-e5-small", "onnx/model_quantized.onnx", "tokenizer.json")

    @cached_property
    def _sesion(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        opciones = ort.SessionOptions()
        # Sin la reserva de memoria (arena) de ONNX Runtime: algo más lento, bastante menos RAM.
        opciones.enable_cpu_mem_arena = False
        ruta = hf_hub_download(self.nombre, self.archivo_modelo)
        return ort.InferenceSession(ruta, opciones, providers=["CPUExecutionProvider"])

    @cached_property
    def _tokenizador(self):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        ruta = hf_hub_download(self.nombre, self.archivo_tokenizador)
        tokenizador = Tokenizer.from_file(ruta)
        tokenizador.enable_truncation(max_length=512)
        tokenizador.enable_padding()
        return tokenizador

    def pasajes(self, textos: list[str]) -> np.ndarray:
        return self._codificar([f"passage: {t}" for t in textos])

    def consulta(self, texto: str) -> np.ndarray:
        return self._codificar([f"query: {texto}"])[0]

    def _codificar(self, textos: list[str]) -> np.ndarray:
        vectores = [self._lote(textos[i : i + self.lote]) for i in range(0, len(textos), self.lote)]
        return np.concatenate(vectores).astype(np.float32)

    def _lote(self, textos: list[str]) -> np.ndarray:
        codificados = self._tokenizador.encode_batch(textos)
        ids = np.array([c.ids for c in codificados], dtype=np.int64)
        mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)
        (tokens,) = self._sesion.run(
            None,
            {"input_ids": ids, "attention_mask": mascara, "token_type_ids": np.zeros_like(ids)},
        )
        # Mean pooling: suma de los vectores de los tokens reales / número de tokens reales.
        peso = mascara[..., None].astype(np.float32)
        medias = (tokens * peso).sum(axis=1) / np.clip(peso.sum(axis=1), 1e-9, None)
        return medias / np.linalg.norm(medias, axis=1, keepdims=True)


@dataclass
class E5Local:
    """Modelo E5 de sentence-transformers (PyTorch). Se usó hasta pasar a ONNX; sigue
    disponible para comparar (`uv sync --extra torch`)."""

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


def modelo_configurado() -> E5Onnx:
    """El modelo que indica la variable de entorno `RAG_EMBEDDINGS`: `fp32` (por defecto,
    mejor calidad) o `int8` (unos 330 MB menos de memoria, 2 puntos menos de recall@5 en la
    búsqueda híbrida). Permite cambiarlo en el servidor sin tocar el código."""
    import os

    variante = os.getenv("RAG_EMBEDDINGS", "fp32").lower()
    if variante == "int8":
        return E5Onnx.int8()
    if variante != "fp32":
        raise ValueError(f"RAG_EMBEDDINGS debe ser fp32 o int8, no {variante!r}")
    return E5Onnx()


def main() -> None:
    modelo = modelo_configurado()
    for estrategia in ("articulo", "fijo"):
        ruta = indexar(estrategia, modelo)
        vectores, _ = cargar_indice(estrategia, modelo.nombre)
        print(
            f"{estrategia:9} {vectores.shape[0]} vectores de {vectores.shape[1]} números en {ruta}"
        )


if __name__ == "__main__":
    main()
