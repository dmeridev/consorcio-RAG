"""Demo en Streamlit (para Streamlit Community Cloud, gratuito).

La lógica (búsqueda, caché, ejemplos, degradación controlada) es la de `rag_consorcio.demo`;
aquí solo está la interfaz. En Streamlit Community Cloud, la clave se guarda en los
*secrets* de la app (`GROQ_API_KEY = "..."`), que Streamlit expone como variable de entorno.

Uso local: `uv run streamlit run streamlit_app.py`
"""

import streamlit as st

from rag_consorcio.demo import (
    AVISO_LEGAL,
    MAX_CARACTERES_PREGUNTA,
    PREGUNTAS_EJEMPLO,
    Demo,
    preparar_indices,
)

st.set_page_config(page_title="Consorcio RAG", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Preparando el buscador (solo la primera vez)…")
def cargar_demo() -> Demo:
    """Una sola instancia para todos los usuarios: el modelo, el índice y la caché de
    respuestas se cargan una vez y se comparten."""
    preparar_indices()
    demo = Demo()
    demo.calentar()
    return demo


demo = cargar_demo()

st.title("Consulta la normativa del Consorcio de Compensación de Seguros")
st.markdown(
    "Pregunta en lenguaje normal. La respuesta sale del **Estatuto Legal del Consorcio**, el "
    "**Reglamento del seguro de riesgos extraordinarios** y la **Ley de Contrato de Seguro** "
    "(textos consolidados del BOE), y cita el artículo."
)
st.warning(AVISO_LEGAL.replace("⚠️ ", ""), icon="⚠️")

if "pregunta" not in st.session_state:
    st.session_state.pregunta = ""

st.caption("Prueba con un ejemplo:")
columnas = st.columns(2)
for i, ejemplo in enumerate(PREGUNTAS_EJEMPLO):
    if columnas[i % 2].button(ejemplo, key=f"ejemplo_{i}", use_container_width=True):
        st.session_state.pregunta = ejemplo

with st.form("consulta"):
    pregunta = st.text_area(
        "Tu pregunta",
        key="pregunta",
        max_chars=MAX_CARACTERES_PREGUNTA,
        placeholder="Por ejemplo: ¿cubre el Consorcio los daños de una inundación?",
    )
    enviada = st.form_submit_button("Preguntar", type="primary", use_container_width=True)

if (enviada or pregunta) and pregunta.strip():
    with st.spinner("Buscando en la normativa y redactando la respuesta…"):
        respuesta, fragmentos = demo.responder(pregunta)
    st.markdown(respuesta)
    if fragmentos:
        with st.expander("Artículos encontrados por la búsqueda"):
            st.markdown(fragmentos, unsafe_allow_html=True)

st.divider()
st.caption(
    "Búsqueda híbrida (embeddings locales con ONNX Runtime + BM25, combinados con Reciprocal "
    "Rank Fusion) y respuesta generada con `gpt-oss-20b` en Groq. "
    "Código: [github.com/dmeridev/consorcio-RAG](https://github.com/dmeridev/consorcio-RAG)."
)
