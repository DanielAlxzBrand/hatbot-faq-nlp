"""Entry point de Hatbot — navegación multi-página y página de chat conversacional."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.nlp_engine import retrieve_relevant_knowledge

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Estado de sesión compartido
# ---------------------------------------------------------------------------
SESSION_DEFAULTS: dict = {
    "messages":    [],
    "last_result": None,
    "debug_mode":  False,
    "top_k":       5,
    "query_count": 0,
}


def _init_session() -> None:
    """Inicializa las claves de session_state con sus valores por defecto."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Helpers de renderizado del chat
# ---------------------------------------------------------------------------

def _source_badge(source: str) -> str:
    """Retorna HTML de badge coloreado según la fuente del resultado."""
    if source == "hardware":
        return (
            '<span style="background:#1d4ed8;color:#fff;padding:2px 8px;'
            'border-radius:4px;font-size:0.75rem;font-weight:600;">🔧 Hardware</span>'
        )
    return (
        '<span style="background:#15803d;color:#fff;padding:2px 8px;'
        'border-radius:4px;font-size:0.75rem;font-weight:600;">🎫 Ticket</span>'
    )


def _render_single_result(result: dict) -> None:
    """Renderiza una tarjeta visual para un resultado individual (HW o ticket)."""
    st.markdown(_source_badge(result["source"]), unsafe_allow_html=True)
    st.markdown(f"**{result['title']}**")

    score = min(max(float(result["score"]), 0.0), 1.0)
    st.progress(score, text=f"Similitud: {score:.1%}")

    summary = result.get("summary") or ""
    if len(summary) > 300:
        st.caption(summary[:300] + "…")
        with st.expander("📄 Ver resumen completo"):
            st.write(summary)
    else:
        st.caption(summary)

    with st.expander("Ver detalles"):
        meta = result.get("metadata") or {}
        if meta:
            for field, value in meta.items():
                if isinstance(value, list):
                    st.markdown(f"**{field}**")
                    for item in value:
                        st.markdown(f"- {item}")
                elif isinstance(value, dict):
                    st.markdown(f"**{field}**")
                    st.json(value)
                else:
                    col_a, col_b = st.columns([1, 3])
                    col_a.markdown(f"**{field}**")
                    col_b.write(value)
        else:
            st.info("Sin metadatos adicionales.")


def render_assistant_response(response: dict) -> None:
    """Renderiza la respuesta completa del asistente dentro de un chat_message."""
    hw_results     = response.get("hardware_results", [])
    ticket_results = response.get("ticket_results", [])
    error          = response.get("error")

    if error:
        st.error(
            f"⚠️ El motor NLP devolvió un error: `{error}`\n\n"
            "Intenta reformular tu pregunta o recarga la página."
        )
        return

    if not hw_results and not ticket_results:
        st.warning(
            "🤔 No encontré resultados para esa consulta.\n\n"
            "**Sugerencias:**\n"
            "- Usa términos técnicos más específicos *(ej: sobrecalentamiento, pantalla azul)*\n"
            "- Menciona el modelo del equipo si lo conoces\n"
            "- Aumenta el número de resultados en el sidebar"
        )
        return

    st.caption(
        f"⏱️ {response['search_time_ms']:.0f} ms  ·  "
        f"{response['total_results']} resultado(s) encontrado(s)"
    )

    if hw_results:
        st.markdown("### 🔧 Soluciones de Hardware")
        for result in hw_results:
            with st.container(border=True):
                _render_single_result(result)

    if hw_results and ticket_results:
        st.divider()

    if ticket_results:
        st.markdown("### 🎫 Tickets Similares")
        for result in ticket_results:
            with st.container(border=True):
                _render_single_result(result)

    if st.session_state["debug_mode"]:
        with st.expander("🔍 Respuesta cruda del motor NLP"):
            st.json(response)


# ---------------------------------------------------------------------------
# Página de chat
# ---------------------------------------------------------------------------

def _run_chat() -> None:
    """Ejecuta la página de chat conversacional."""
    _init_session()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        st.slider("Resultados por fuente", min_value=1, max_value=10, key="top_k")
        st.toggle("🔍 Modo Debug", key="debug_mode")
        st.divider()
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state["messages"]    = []
            st.session_state["last_result"] = None
            st.rerun()
        st.metric("Consultas en sesión", st.session_state["query_count"])
        st.divider()
        st.caption("Hatbot FAQ NLP · Fase 3")

    # Encabezado
    st.markdown("# DB Axion AI Support")
    st.markdown(
        "Describe tu problema de hardware o IT y el motor NLP buscará "
        "soluciones en la knowledge base y en tickets históricos resueltos."
    )
    st.divider()

    # Historial del chat
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant_response(msg["content"])

    # Capturar nueva consulta
    if prompt := st.chat_input("Describe tu problema de hardware o IT..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["query_count"] += 1

        with st.spinner("🔍 Buscando soluciones…"):
            try:
                response = retrieve_relevant_knowledge(
                    prompt,
                    top_k=st.session_state["top_k"],
                )
            except Exception as exc:
                response = {
                    "query":            prompt,
                    "hardware_results": [],
                    "ticket_results":   [],
                    "total_results":    0,
                    "search_time_ms":   0.0,
                    "error":            str(exc),
                }

        st.session_state["last_result"] = response
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.rerun()


# ---------------------------------------------------------------------------
# Navegación multi-página
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="DB Axion AI Support",
        page_icon="🤖",
        layout="wide",
    )

    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])

    if _ver >= (1, 36):
        from app.dashboard_ui import run as _run_dashboard  # noqa: E402

        _pages = st.navigation(
            [
                st.Page(_run_chat,      title="Chat",      icon="💬", default=True),
                st.Page(_run_dashboard, title="Dashboard", icon="📊"),
            ],
            position="sidebar",
        )
        _pages.run()

    else:
        # Fallback para Streamlit < 1.36
        _choice = st.sidebar.radio("Página", ["💬 Chat", "📊 Dashboard"])
        if _choice == "💬 Chat":
            _run_chat()
        else:
            from app.dashboard_ui import run as _run_dashboard  # noqa: E402
            _run_dashboard()
