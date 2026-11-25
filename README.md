# consorcio-RAG

Asistente que responde preguntas sobre la **normativa del Consorcio de Compensación de
Seguros** citando el artículo exacto del que sale cada respuesta.

> **Pregunta:** ¿Hay plazo de carencia si sustituyo mi póliza por otra en otra compañía?
>
> **Respuesta:** No, siempre que no haya interrupción de la cobertura. El plazo de carencia
> de 7 días no rige en los casos de reemplazo o sustitución de la póliza sin solución de
> continuidad, salvo en la parte que sea aumento o nueva cobertura.
> *Fuente: Reglamento del seguro de riesgos extraordinarios, art. 8.1.a.*

**Estado:** en desarrollo. La recuperación está terminada y evaluada: el artículo correcto
está entre los 5 primeros resultados en el **80 %** de las preguntas. Faltan la generación
de la respuesta y la demo pública.

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
| Búsqueda por palabras (BM25) | `bm25.py` | ✅ |
| Búsqueda híbrida (RRF y suma ponderada) | `buscar.py` | ✅ |
| Evaluación de la recuperación (recall@k, MRR) | `evaluar.py` | ✅ |
| Evaluación de la respuesta (fidelidad) | | ⏳ |
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

**BM25 implementado a mano.** Los embeddings captan el significado pero casi ignoran los
datos exactos («artículo 9»); BM25 hace lo contrario. Está implementado en unas 40 líneas
(variante de Lucene, `k1 = 1,5`, `b = 0,75`) y un test lo compara con la librería `bm25s`
sobre el corpus real. El tokenizador pasa a minúsculas y quita tildes y palabras vacías,
pero conserva la ñ, los números y «no». La reducción a raíces (Snowball) es opcional y se
decidirá con la evaluación.

**Búsqueda híbrida con Reciprocal Rank Fusion.** Las puntuaciones de embeddings
(0,80-0,90) y BM25 (0-15) no se pueden sumar tal cual. RRF solo mira la posición de cada
fragmento en cada lista, `1 / (60 + posición)`, así que no hay nada que ajustar. Como
alternativa para comparar hay una suma ponderada con normalización min-max. En una primera
prueba con 5 preguntas, cada buscador por separado falla en alguna, mientras que las dos
híbridas aciertan todas a la primera. La evaluación decidirá entre ellas.

| Estrategia | Fragmentos | Mediana | Mezclan varios artículos |
|---|---|---|---|
| Por artículo | 215 | 751 caracteres | 0 |
| Tamaño fijo (1.000 / 200) | 219 | 996 caracteres | 129 |

## Evaluación de la recuperación

63 preguntas generadas con un LLM a partir de los artículos (59 con respuesta en el corpus),
cada una con su lista de artículos válidos y un **tipo** para controlar el sesgo léxico:
paráfrasis, literal, caso práctico y referencia a un artículo. Se prueban 14
configuraciones: 2 troceados × 4 métodos × con o sin reducción a raíces.

| Configuración | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| **Por artículo · RRF · raíces** | **58 %** | **80 %** | **0,65** |
| Por artículo · embeddings | 47 % | 78 % | 0,61 |
| Por artículo · RRF sin raíces | 53 % | 76 % | 0,63 |
| Por artículo · ponderada · raíces | 54 % | 76 % | 0,63 |
| Por artículo · BM25 · raíces | 42 % | 75 % | 0,56 |
| Por artículo · BM25 | 46 % | 64 % | 0,54 |

- **La búsqueda híbrida gana:** 80 % de recall@5 frente al 64 % de BM25 solo.
- **Las raíces mejoran BM25** del 64 % al 75 %.
- **Troceado por artículo vs. fijo:** empatan en el global (80 %), pero en las preguntas que
  citan un artículo por número BM25 acierta el 100 % por artículo y el 20 % con fragmentos
  fijos, gracias a la cabecera de contexto.
- **Análisis de errores:** casi todos los fallos se deben a la distancia entre el lenguaje
  coloquial y el jurídico («ha quebrado» frente a «declarada en concurso»).

**Limitaciones:** las preguntas son sintéticas y no se han revisado a mano, y 59 preguntas
son pocas (cada una vale 1,7 puntos). Solo las diferencias grandes son concluyentes.

## Uso

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m rag_consorcio.trocear       # PDF → fragmentos (data/fragmentos/)
uv run python -m rag_consorcio.embeddings    # fragmentos → vectores (data/indices/)
uv run python -m rag_consorcio.evaluar       # evaluación de la recuperación (~2,5 min)
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
  bm25.py                 búsqueda por palabras y tokenizador
  buscar.py               búsqueda híbrida (RRF y suma ponderada)
  evaluar.py              métricas y evaluación de todas las configuraciones
data/evaluacion/          preguntas de evaluación y resultados
tests/
```

## Cómo se ha construido

Desarrollado con [Claude Code](https://claude.com/claude-code) como agente de
programación. Las decisiones de arquitectura se han tomado comparando alternativas y
validándolas con medidas y tests.
