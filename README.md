# consorcio-RAG

[![CI](https://github.com/dmeridev/consorcio-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/dmeridev/consorcio-RAG/actions/workflows/ci.yml)

Asistente que responde preguntas sobre la **normativa del Consorcio de Compensación de
Seguros** citando el artículo exacto del que sale cada respuesta.

**▶ [Prueba la demo](https://consorcio-rag-7mdqdogulktgay6rkan6xf.streamlit.app/)**
(si lleva tiempo sin visitas, tarda un minuto en despertar).

> **Pregunta:** ¿Hay plazo de carencia si sustituyo mi póliza por otra en otra compañía?
>
> **Respuesta:** No, siempre que no haya interrupción de la cobertura. El plazo de carencia
> de 7 días no rige en los casos de reemplazo o sustitución de la póliza sin solución de
> continuidad, salvo en la parte que sea aumento o nueva cobertura.
> *Fuente: Reglamento del seguro de riesgos extraordinarios, art. 8.1.a.*

**Estado:** la recuperación está evaluada (el artículo correcto está entre los 5 primeros
resultados en el **80 %** de las preguntas), la respuesta se genera con citas verificadas y
su fidelidad está medida, la demo está publicada y cada cambio pasa por CI con la
evaluación como test.

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
| Evaluación de la respuesta (fidelidad, LLM como juez) | `fidelidad.py` | ✅ |
| Generación de la respuesta con citas verificadas | `generar.py` | ✅ |
| Demo web (Streamlit, y Gradio opcional) | `demo.py`, `streamlit_app.py` | ✅ |
| Publicación en Streamlit Community Cloud | [demo](https://consorcio-rag-7mdqdogulktgay6rkan6xf.streamlit.app/) | ✅ |
| CI/CD con GitHub Actions (evaluación como test, despliegue si pasa) | `.github/workflows/ci.yml` | ✅ |
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
- **Se ejecuta con ONNX Runtime, sin PyTorch**, para que la demo quepa en un hosting
  gratuito. El mean pooling y la normalización se hacen a mano. Los vectores son idénticos
  a los de sentence-transformers (coseno 1,000000) y la evaluación da las mismas cifras,
  con la memoria bajando de ~1.045 a ~813 MB. Una versión cuantizada a int8
  (`RAG_EMBEDDINGS=int8`) baja a ~480 MB a cambio de 2 puntos de recall@5 en la búsqueda
  híbrida.

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

## Generación de la respuesta

Los 5 fragmentos recuperados se etiquetan ([F1]…[F5]) y se pasan al LLM con instrucciones
de responder solo con ellos. El modelo devuelve **JSON con un esquema estricto**
(`respuesta`, `citas`, `respondida`) y **el código verifica** que cada cita es uno de los
fragmentos recibidos: una instrucción no es una garantía.

- **Proveedor intercambiable:** una sola clase para cualquier API compatible con OpenAI
  (Groq, OpenAI, Gemini). Por defecto, el plan gratuito de Groq (`openai/gpt-oss-120b`).
- **Degradación controlada:** si el LLM falla, se devuelven los artículos encontrados.
- **Limitación conocida:** que una cita exista no garantiza que la afirmación esté en el
  artículo citado (*faithfulness*). Por eso se mide aparte.

Requiere una clave en `.env`:

```
GROQ_API_KEY=...
```

## Evaluación de la respuesta (fidelidad)

Un **LLM como juez**, distinto del generador para evitar el sesgo de autopreferencia
(`qwen/qwen3.8-27b` juzga a `gpt-oss-120b`), parte cada respuesta en afirmaciones y
comprueba cada una **solo contra los fragmentos citados**, como hace RAGAS.

| Métrica | Resultado |
|---|---|
| Afirmaciones respaldadas | 89 % |
| Respuestas totalmente fieles | 73 % |
| Tasa de respuesta (preguntas con respuesta) | 86 % |
| Abstención correcta (preguntas sin respuesta) | 3 de 4 |

**Las cifras son orientativas.** El juez se calibró con dos casos de verdad conocida y
falló en uno: aceptó como fiel una respuesta que deducía que el granizo está cubierto
porque el art. 4 del Reglamento menciona «eventos de la naturaleza», que ahí es un ramo de
seguro y no un fenómeno cubierto. La revisión manual de los errores muestra que el juez es
fiable con lo literal y débil con las interpretaciones forzadas, y que el problema
principal está en el generador: **cuando la búsqueda no trae el artículo correcto, tiende a
estirar el texto** en lugar de abstenerse.

### Instrucciones v2: no deducir, abstenerse si no hay respuesta expresa

Para atacar ese problema, la v2 de las instrucciones prohíbe deducir (que algo no esté
excluido no significa que esté cubierto; que una palabra coincida no significa que el
fragmento hable de lo mismo) y obliga a abstenerse si ningún fragmento responde de forma
expresa. Son reglas generales: no se añadió nada específico de las preguntas de la
evaluación, para no sobreajustar a ella.

Primera comparación sobre las 34 preguntas evaluadas con las dos versiones (la v2 está
incompleta porque se agotó la cuota diaria del plan gratuito):

| Métrica | v1 | v2 |
|---|---|---|
| Afirmaciones respaldadas | 93 % | 95 % |
| Tasa de respuesta | 85 % | 74 % |
| Responde aunque la búsqueda no trajo el artículo | 38 % | **12 %** |

La v2 deja de inventar cuando le falta el artículo (desaparece el error del granizo), a
cambio de responder menos y de pasarse de prudente en algún caso. Es el intercambio entre
precisión y cobertura, que en un dominio legal compensa.

Las respuestas y los juicios se guardan en caché (`data/evaluacion/`), distinguiendo la
versión de las instrucciones, así que repetir la evaluación no consume cuota. El plan
gratuito de Groq limita los tokens por minuto (una ejecución completa tarda unos 30
minutos) y por día (200.000 tokens, en torno a una evaluación completa).

## Demo web

Una interfaz en Streamlit (`streamlit_app.py`), pensada para Streamlit Community Cloud: la
pregunta, la respuesta con sus fuentes, los artículos encontrados (desplegables, con el
citado marcado) y un aviso de que es una demo educativa que no sustituye el asesoramiento
profesional. Hay también una versión en Gradio (`app.py`, opcional).

Está pensada para funcionar con una cuota gratuita limitada (unas 80-100 respuestas al día):

- Modelo ligero (`gpt-oss-20b`), con una cuota separada de la de las evaluaciones.
- Caché de respuestas y ejemplos precalculados, que no consumen cuota.
- Si el LLM no responde o se agota la cuota, se muestran igualmente los artículos encontrados.
- El modelo de embeddings se precarga al arrancar: una pregunta nueva tarda en torno a 1-2
  segundos.

```bash
uv run streamlit run streamlit_app.py      # http://localhost:8501
uv sync --extra gradio && uv run python app.py   # versión Gradio, http://localhost:7860
```

En Streamlit Community Cloud la clave va en los *secrets* de la app (`GROQ_API_KEY`), y
`RAG_EMBEDDINGS = "int8"` reduce la memoria si hiciera falta.

## Uso

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m rag_consorcio.trocear       # PDF → fragmentos (data/fragmentos/)
uv run python -m rag_consorcio.embeddings    # fragmentos → vectores (data/indices/)
uv run python -m rag_consorcio.evaluar       # evaluación de la recuperación (~2,5 min)
uv run python -m rag_consorcio.generar "¿Hay carencia si cambio de compañía?"
uv run python -m rag_consorcio.fidelidad     # fidelidad con un LLM como juez (~30 min la 1.ª vez)
uv run pytest                                # todos los tests
uv run pytest -m "not modelo"                # sin los que cargan el modelo de embeddings
```

La primera ejecución de `embeddings` descarga el modelo ONNX (unos 450 MB). Para comparar con
PyTorch: `uv sync --extra torch`.

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
  generar.py              respuesta del LLM con citas verificadas
  fidelidad.py            evaluación de la respuesta con un LLM como juez
  demo.py                 lógica de la demo (caché, ejemplos, degradación)
streamlit_app.py          demo en Streamlit
app.py                    demo en Gradio (opcional)
requirements.txt          dependencias para Streamlit Community Cloud
data/evaluacion/          preguntas de evaluación y resultados
data/demo/                respuestas precalculadas de los ejemplos
tests/
.github/workflows/ci.yml  CI/CD
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) comprueba cada cambio con tres trabajos en
paralelo:

1. **Estilo:** `ruff check` y `ruff format --check`.
2. **Tests rápidos:** los que no necesitan el modelo de embeddings.
3. **Tests con el modelo y evaluación de la búsqueda.** La evaluación funciona como un
   test: **si el recall@5 de la configuración principal baja del 78 %** (hoy, 79,7 %), el
   CI falla igual que si se rompiera una función. El modelo se guarda en caché entre
   ejecuciones.

Si los tres pasan en `main`, el trabajo **Publicar la demo** copia `main` a la rama
`deploy`, que es la que despliega Streamlit Community Cloud. **La demo solo recibe código
que ha pasado los tests y la evaluación.**

La evaluación de fidelidad no está en el CI: necesita la API del LLM, gasta cuota y tarda
unos 30 minutos. Se ejecuta a mano.

## Cómo se ha construido

Desarrollado con [Claude Code](https://claude.com/claude-code) como agente de
programación. Las decisiones de arquitectura se han tomado comparando alternativas y
validándolas con medidas y tests.
