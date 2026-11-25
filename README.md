# consorcio-RAG

Asistente que responde preguntas sobre la **normativa del Consorcio de Compensación de
Seguros** citando el artículo exacto del que sale cada respuesta.

> **Pregunta:** ¿Hay plazo de carencia si sustituyo mi póliza por otra en otra compañía?
>
> **Respuesta:** No, siempre que no haya interrupción de la cobertura. El plazo de carencia
> de 7 días no rige en los casos de reemplazo o sustitución de la póliza sin solución de
> continuidad, salvo en la parte que sea aumento o nueva cobertura.
> *Fuente: Reglamento del seguro de riesgos extraordinarios, art. 8.1.a.*

**Estado:** en desarrollo. Ya funcionan la extracción, el troceado y los embeddings; faltan
la búsqueda híbrida, la evaluación, la generación de la respuesta y la demo pública.

---

## Corpus

Textos **consolidados** del BOE, es decir, con todas sus modificaciones incorporadas:

| Documento | Referencia | Páginas |
|---|---|---|
| Estatuto Legal del Consorcio de Compensación de Seguros (RDLeg 7/2004) | [BOE-A-2004-18910](https://www.boe.es/buscar/act.php?id=BOE-A-2004-18910) | 19 |
| Reglamento del seguro de riesgos extraordinarios (RD 300/2004) | [BOE-A-2004-3373](https://www.boe.es/buscar/act.php?id=BOE-A-2004-3373) | 12 |
| Ley 50/1980 de Contrato de Seguro | [BOE-A-1980-22501](https://www.boe.es/buscar/act.php?id=BOE-A-1980-22501) | 31 |

Solo se usan datos públicos.

## Arquitectura

```
INDEXADO (una vez)
  PDF del BOE → limpieza → troceado → embeddings → índice

CONSULTA (cada pregunta)
  pregunta → embedding → fragmentos más parecidos → LLM → respuesta con citas
```

| Pieza | Módulo | Estado |
|---|---|---|
| Extracción y limpieza | `extraer.py` | ✅ |
| Troceado (por artículo y de tamaño fijo) | `trocear.py` | ✅ |
| Embeddings locales | `embeddings.py` | ✅ |
| Búsqueda híbrida (embeddings + BM25) | | ⏳ |
| Evaluación (recall@k, fidelidad de la respuesta) | | ⏳ |
| Generación de la respuesta con citas | | ⏳ |
| Demo pública (Hugging Face Spaces) | | ⏳ |
| CI/CD (GitHub Actions) | | ⏳ |

## Decisiones de diseño

**Limpieza del PDF.** Se descartan el índice inicial y las cabeceras y pies de página
(«BOLETÍN OFICIAL DEL ESTADO», «Página N»). El índice repite los títulos de los artículos
y, si se queda, la búsqueda devuelve la línea del índice en lugar del artículo.

**Troceado por artículo, con troceado fijo para comparar.**

- Un fragmento por artículo o disposición. Los largos se parten por apartados numerados,
  sin partir nunca un párrafo.
- Cada fragmento empieza con una **cabecera de contexto** (norma, artículo, título y
  capítulo), para que ninguno pierda de qué trata aunque sea la segunda parte de un
  artículo largo.
- Como referencia, también se generan fragmentos de 1.000 caracteres con 200 de solape.
  El 59 % de ellos mezcla varios artículos. La evaluación medirá cuál recupera mejor el
  artículo correcto.

**Norma aprobatoria y texto aprobado.** El PDF del Reglamento contiene primero el Real
Decreto que lo aprueba y después el Reglamento. Las dos partes repiten nombres de
disposiciones, así que cada fragmento sabe a cuál pertenece y se cita en consecuencia.

**Artículos derogados fuera.** Las unidades que solo dicen «(Derogado)» no se indexan:
harían de cebo en la búsqueda.

**Embeddings locales: `intfloat/multilingual-e5-small`.**

- Gratuito, multilingüe y ejecutado en CPU: la demo pública no depende de ninguna API de
  pago y los textos no salen del equipo.
- El modelo es **intercambiable** (`ModeloEmbeddings` es un `Protocol` con dos métodos).
- El modelo solo lee 512 tokens y corta el resto sin avisar. Con el límite inicial de
  3.000 caracteres, 17 fragmentos se pasaban. Se bajó a 2.000 (máximo 466 tokens) y un
  test lo vigila.

| Estrategia | Fragmentos | Mediana | Mezclan varios artículos |
|---|---|---|---|
| Por artículo | 215 | 751 caracteres | 0 |
| Tamaño fijo (1.000 / 200) | 219 | 996 caracteres | 129 |

## Uso

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m rag_consorcio.trocear       # PDF → fragmentos (data/fragmentos/)
uv run python -m rag_consorcio.embeddings    # fragmentos → vectores (data/indices/)
uv run pytest                                # todos los tests
uv run pytest -m "not modelo"                # sin los que cargan el modelo de embeddings
```

La primera ejecución de `embeddings` descarga el modelo (unos 470 MB).

## Estructura

```
data/raw/                 PDFs del BOE
src/rag_consorcio/
  corpus.py               documentos y cómo se cita cada uno
  extraer.py              PDF → texto limpio → artículos y disposiciones
  numeros.py              «treinta y tres a)» → «33 a)»
  trocear.py              las dos estrategias de troceado
  embeddings.py           vectores con un modelo local intercambiable
tests/
```

## Cómo se ha construido

Desarrollado con [Claude Code](https://claude.com/claude-code) como agente de
programación. Las decisiones de arquitectura se han tomado comparando alternativas y
validándolas con medidas y tests.
