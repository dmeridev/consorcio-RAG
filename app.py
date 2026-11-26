"""Punto de entrada de la demo (Hugging Face Spaces busca este fichero)."""

from rag_consorcio.demo import Demo, construir_interfaz, preparar_indices

preparar_indices()
demo = Demo()
demo.calentar()
interfaz = construir_interfaz(demo)

if __name__ == "__main__":
    interfaz.launch()
