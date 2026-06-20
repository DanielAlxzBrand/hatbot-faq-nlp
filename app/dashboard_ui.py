"""Dashboard analítico — KPIs, gráficos y explorador de la Knowledge Base de hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processor import PATHS, get_dataframe_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carga de datos con caché
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_stats() -> dict:
    """Carga el Parquet completo y retorna estadísticas enriquecidas para los gráficos."""
    parquet_path = Path(PATHS["clean_parq"])
    try:
        df = pd.read_parquet(parquet_path)
        stats = get_dataframe_stats(df)
        stats["priority_dist"]  = df["priority"].value_counts().to_dict()
        stats["segment_dist"]   = df["customer_segment"].value_counts().to_dict()
        stats["sentiment_dist"] = df["customer_sentiment"].value_counts().to_dict()
        stats["channel_dist"]   = df["channel"].value_counts().to_dict()
        logger.info("Estadísticas cargadas: %d tickets desde %s", stats["total_tickets"], parquet_path)
        return stats
    except FileNotFoundError:
        logger.exception("Parquet no encontrado: %s", parquet_path.resolve())
        st.error(
            f"⚠️ No se encontró el dataset procesado.\n\n"
            f"Ruta esperada: `{parquet_path.resolve()}`\n\n"
            "Ejecuta `python src/data_processor.py` para generarlo."
        )
        return {}
    except Exception:
        logger.exception("Error inesperado al cargar estadísticas desde %s", parquet_path)
        st.error("⚠️ Error inesperado al cargar las estadísticas del dataset.")
        return {}


@st.cache_data(ttl=3600)
def load_hardware_kb() -> list[dict]:
    """Carga las entradas de la Knowledge Base de hardware desde JSON."""
    hw_path = Path(PATHS["hw_kb"])
    try:
        with open(hw_path, encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("hardware_entries", [])
        logger.info("Knowledge Base cargada: %d entradas desde %s", len(entries), hw_path)
        return entries
    except FileNotFoundError:
        logger.exception("Knowledge Base no encontrada: %s", hw_path.resolve())
        st.error(
            f"⚠️ No se encontró la Knowledge Base.\n\n"
            f"Ruta esperada: `{hw_path.resolve()}`\n\n"
            "Ejecuta `python src/data_processor.py` para generarla."
        )
        return []
    except json.JSONDecodeError:
        logger.exception("JSON inválido en Knowledge Base: %s", hw_path)
        st.error("⚠️ El archivo de Knowledge Base tiene formato JSON inválido.")
        return []
    except KeyError:
        logger.exception("Clave 'hardware_entries' ausente en %s", hw_path)
        st.error("⚠️ La Knowledge Base no tiene la clave esperada 'hardware_entries'.")
        return []


# ---------------------------------------------------------------------------
# Helpers de gráficos
# ---------------------------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    margin=dict(l=0, r=0, t=30, b=0),
    height=320,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def _bar_chart_plotly_h(data: dict, title: str) -> None:
    """Renderiza un bar chart horizontal Plotly con datos {label: count}."""
    labels = list(data.keys())
    values = list(data.values())
    fig = px.bar(
        x=values, y=labels, orientation="h",
        title=title, labels={"x": "Tickets", "y": ""},
        color=values, color_continuous_scale="Blues",
    )
    fig.update_layout(**_PLOTLY_LAYOUT, coloraxis_showscale=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def _pie_chart_plotly(data: dict, title: str) -> None:
    """Renderiza un gráfico de tipo donut (pie con agujero central) usando Plotly.

    Utiliza un valor de ``hole=0.45`` para lograr un efecto visual de donut
    reconocible, manteniendo un buen equilibrio estético.
    """
    fig = px.pie(
        names=list(data.keys()),
        values=list(data.values()),
        title=title,
        hole=0.45,  # Valor ajustado para que se visualice claramente como donut
    )
    fig.update_layout(**_PLOTLY_LAYOUT)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def _bar_chart_plotly_vertical(data: dict, title: str) -> None:
    """Renderiza un bar chart vertical con Plotly Express.

    Equivalente visual a la versión Altair eliminada.
    Úsalo para distribuciones de segmento y sentimiento.
    """
    labels = list(data.keys())
    values = list(data.values())
    fig = px.bar(
        x=labels,
        y=values,
        title=title,
        labels={"x": "", "y": "Tickets"},
        color=values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(**_PLOTLY_LAYOUT, coloraxis_showscale=False)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Secciones de renderizado
# ---------------------------------------------------------------------------

def render_kpis(stats: dict) -> None:
    """Renderiza la fila de métricas KPI principales."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Tickets",           f"{stats['total_tickets']:,}")
    c2.metric("🔓 Tickets Abiertos",        f"{stats['open_tickets']:,}")
    c3.metric("✅ Tickets Cerrados",         f"{stats['closed_tickets']:,}")
    avg = stats["avg_resolution_hours"]
    c4.metric("⏱️ Tiempo Prom. Resolución", f"{avg}h" if avg else "N/A")


def render_date_range(stats: dict) -> None:
    """Renderiza el rango de fechas y métricas secundarias del dataset."""
    dr = stats.get("date_range", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Fecha más antigua",  dr.get("min", "—"))
    c2.metric("📅 Fecha más reciente", dr.get("max", "—"))
    c3.metric("💾 Tamaño en memoria",  f"{stats.get('memory_mb', 0):.1f} MB")


def render_charts(stats: dict) -> None:
    """Renderiza las dos filas de gráficos analíticos (100% Plotly)."""
    st.markdown("### 📈 Distribución de Tickets")

    col_l, col_r = st.columns(2)
    with col_l:
        top10 = dict(
            sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        _bar_chart_plotly_h(top10, "Top 10 tipos de incidencia")
    with col_r:
        _pie_chart_plotly(stats["priority_dist"], "Distribución por prioridad")

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        _bar_chart_plotly_vertical(stats["segment_dist"], "Segmento de cliente")
    with col_r2:
        _bar_chart_plotly_vertical(stats["sentiment_dist"], "Sentimiento del cliente")


def _matches_search(entry: dict, term: str) -> bool:
    """Retorna True si la entrada coincide con el término de búsqueda (case-insensitive)."""
    if not term:
        return True
    haystack = " ".join([
        entry.get("problema", ""),
        entry.get("modelo_equipo", ""),
        " ".join(entry.get("keywords", [])),
        entry.get("categoria", ""),
    ]).lower()
    return term.lower().strip() in haystack


def _render_entry_card(entry: dict) -> None:
    """Renderiza una entrada de la Knowledge Base dentro de un st.expander."""
    nivel = entry.get("nivel_dificultad", "")
    badge = {"Básico": "🟢", "Intermedio": "🟡", "Avanzado": "🔴"}.get(nivel, "⚪")

    with st.expander(
        f"{entry['id']} — {entry.get('modelo_equipo') or entry.get('problema', '(sin modelo)')} "
        f"· {badge} {nivel}"
    ):
        st.markdown(f"#### {entry['problema']}")

        col_meta, col_sint, col_herr = st.columns(3)

        with col_meta:
            st.markdown("**Categoría**")
            st.write(entry.get("categoria") or "—")
            st.markdown("**Dificultad**")
            st.write(f"{badge} {nivel}" if nivel else "—")
            st.markdown("**Tiempo estimado**")
            mins = entry.get("tiempo_estimado_minutos", 0)
            st.write(f"{mins} min" if mins else "—")

        with col_sint:
            st.markdown("**Síntomas**")
            sintomas = entry.get("sintomas", [])
            if sintomas:
                for s in sintomas:
                    st.markdown(f"- {s}")
            else:
                st.caption("Sin información disponible.")

        with col_herr:
            st.markdown("**Herramientas necesarias**")
            herramientas = entry.get("herramientas_necesarias", [])
            if herramientas:
                for h in herramientas:
                    st.markdown(f"- {h}")
            else:
                st.caption("Sin información disponible.")

        st.divider()

        tab_diag, tab_sol, tab_prev = st.tabs(["🔎 Diagnóstico", "🛠️ Solución", "🛡️ Prevención"])

        with tab_diag:
            pasos = entry.get("pasos_diagnostico", [])
            if pasos:
                for i, paso in enumerate(pasos, 1):
                    st.markdown(f"{i}. {paso}")
            else:
                st.caption("Sin pasos de diagnóstico disponibles.")

        with tab_sol:
            pasos = entry.get("pasos_solucion", [])
            if pasos:
                for i, paso in enumerate(pasos, 1):
                    st.markdown(f"{i}. {paso}")
            else:
                st.caption("Sin pasos de solución disponibles.")

        with tab_prev:
            prev = entry.get("prevencion", [])
            if prev:
                for p in prev:
                    st.markdown(f"- {p}")
            else:
                st.caption("Sin medidas de prevención disponibles.")


def render_hardware_kb(entries: list[dict]) -> None:
    """Renderiza el explorador interactivo de la Knowledge Base de hardware."""
    st.markdown("### 🔧 Hardware Knowledge Base")

    if not entries:
        st.warning("⚠️ No hay entradas disponibles en la Knowledge Base.")
        return

    search_term = st.text_input(
        "🔍 Buscar en Knowledge Base",
        placeholder="Ej: sobrecalentamiento, Acer, BIOS, USB…",
        key="hw_search",
    )

    visible = [e for e in entries if _matches_search(e, search_term)]

    if search_term and not visible:
        st.warning(
            f" No se encontraron entradas para **'{search_term}'**.\n\n"
            "Intenta con otro término: nombre de equipo, síntoma o herramienta."
        )
        return

    total_label = f"{len(visible)} de {len(entries)} entradas"
    if search_term:
        st.caption(f"Resultados para **'{search_term}'**: {total_label}")
    else:
        st.caption(f"Mostrando {total_label}")

    placeholder = st.empty()
    with placeholder.container():
        for entry in visible:
            _render_entry_card(entry)


# ---------------------------------------------------------------------------
# Punto de entrada de la página (callable para st.navigation en main_chat.py)
# ---------------------------------------------------------------------------

def run() -> None:
    """Ejecuta el dashboard completo; llamada como page callable desde main_chat.py."""
    st.markdown("# Dashboard Analítico")
    st.markdown(
        "Métricas del corpus de tickets IT y explorador interactivo "
        "de la Knowledge Base de hardware."
    )
    st.divider()

    with st.spinner("Cargando estadísticas del dataset…"):
        stats = load_stats()
        if not stats:
            return  # load_stats() ya mostró el st.error correspondiente

    with st.spinner("Cargando Knowledge Base…"):
        kb_entries = load_hardware_kb()
        if not kb_entries:
            return  # load_hardware_kb() ya mostró el st.error correspondiente

    render_kpis(stats)
    st.divider()
    render_date_range(stats)
    st.divider()
    render_charts(stats)
    st.divider()
    render_hardware_kb(kb_entries)
