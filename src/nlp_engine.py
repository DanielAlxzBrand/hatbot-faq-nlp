"""
Fase 2 — Motor NLP: búsqueda semántica y por keywords sobre hardware KB y tickets.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "model_name":             "sentence-transformers/all-MiniLM-L6-v2",
    "top_k":                  5,
    "similarity_threshold":   0.35,
    "hw_embeddings_path":     Path("data/processed/hw_embeddings.npz"),
    "ticket_embeddings_path": Path("data/processed/ticket_embeddings.npz"),
    "hw_kb_path":             Path("data/custom/hardware_knowledge_base.json"),
    "clean_parquet_path":     Path("data/processed/tickets_clean.parquet"),
    "batch_size":             64,
    "force_rebuild":          False,
}

# ---------------------------------------------------------------------------
# SINGLETON DEL MODELO
# ---------------------------------------------------------------------------

class _ModelCache:
    _instance: SentenceTransformer | None = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        """Retorna la instancia del modelo, cargándola solo en el primer acceso."""
        if cls._instance is None:
            logging.info("Cargando modelo: %s", CONFIG["model_name"])
            cls._instance = SentenceTransformer(CONFIG["model_name"])
        return cls._instance


class _TfidfCache:
    """Singleton del vectorizador TF-IDF sobre keywords de hardware.

    Construye y cachea el vectorizador + matriz en el primer acceso.
    Las llamadas subsiguientes retornan la misma instancia sin reconstruir.
    Llamar a ``invalidate()`` si la Knowledge Base cambia en runtime.
    """

    _vectorizer: TfidfVectorizer | None = None
    _matrix: Any | None = None          # scipy sparse matrix
    _entries: list[dict] | None = None

    @classmethod
    def get(cls) -> tuple[TfidfVectorizer, Any, list[dict]]:
        """Retorna ``(vectorizer, tfidf_matrix, entries)``, construyendo si es necesario.

        Returns:
            Tupla con el vectorizador entrenado, la matriz TF-IDF sparse y la lista
            de entradas de la KB — en correspondencia posicional 1:1.
        """
        if cls._vectorizer is None:
            logging.info("Construyendo índice TF-IDF sobre keywords de hardware...")
            entries = load_knowledge_base()
            keyword_docs = [" ".join(e["keywords"]) for e in entries]
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(keyword_docs)
            cls._vectorizer = vectorizer
            cls._matrix     = matrix
            cls._entries    = entries
            logging.info(
                "Índice TF-IDF construido: %d documentos, vocabulario: %d términos.",
                len(entries), len(vectorizer.vocabulary_),
            )
        return cls._vectorizer, cls._matrix, cls._entries

    @classmethod
    def invalidate(cls) -> None:
        """Invalida la caché forzando reconstrucción en el próximo acceso.

        Llamar cuando la Knowledge Base sea actualizada en runtime.
        """
        cls._vectorizer = None
        cls._matrix     = None
        cls._entries    = None
        logging.info("_TfidfCache invalidado — se reconstruirá en el próximo acceso.")


# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------

def load_knowledge_base() -> list[dict]:
    """Carga la base de conocimiento de hardware desde JSON.

    Returns:
        Lista de entradas de hardware (hardware_entries del JSON).

    Raises:
        FileNotFoundError: si el archivo JSON no existe en la ruta configurada.
    """
    path = CONFIG["hw_kb_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base no encontrada: {path.resolve()}. "
            "Ejecuta primero: python src/data_processor.py"
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data["hardware_entries"]
    logging.info("Knowledge base cargada: %d entradas desde %s", len(entries), path)
    return entries


def load_ticket_corpus(columns: list[str] | None = None) -> pd.DataFrame:
    """Carga el corpus de tickets desde Parquet y construye el campo search_text.

    Args:
        columns: columnas a leer; por defecto las 6 necesarias para NLP.

    Returns:
        DataFrame con columna extra ``search_text`` lista para vectorizar.
    """
    if columns is None:
        columns = [
            "ticket_id", "initial_message", "resolution_summary",
            "issue_type", "status", "is_open",
        ]
    path = CONFIG["clean_parquet_path"]
    df = pd.read_parquet(path, columns=columns)
    logging.info("Corpus de tickets cargado: %d filas desde %s", len(df), path)

    # Concatenar campos de texto; NA → cadena vacía antes de unir
    text_parts = [
        df["issue_type"].astype(str).fillna(""),
        df["initial_message"].fillna(""),
        df["resolution_summary"].fillna(""),
    ]
    df["search_text"] = text_parts[0] + " " + text_parts[1] + " " + text_parts[2]
    return df


# ---------------------------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------------------------

def generate_embeddings(texts: list[str], show_progress: bool = True) -> np.ndarray:
    """Vectoriza una lista de textos y normaliza cada vector a norma L2.

    Args:
        texts: textos a codificar.
        show_progress: mostrar barra de progreso de sentence-transformers.

    Returns:
        Matriz float32 de shape ``(len(texts), embedding_dim)`` con vectores L2-normalizados.
    """
    model = _ModelCache.get()
    logging.info("Generando embeddings para %d textos (batch_size=%d)...",
                 len(texts), CONFIG["batch_size"])
    matrix = model.encode(
        texts,
        batch_size=CONFIG["batch_size"],
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalización interna
    )
    return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DE ÍNDICES
# ---------------------------------------------------------------------------

def build_hardware_index() -> tuple[np.ndarray, list[dict]]:
    """Vectoriza la KB de hardware y guarda el índice en disco.

    Returns:
        Tupla ``(matrix, entries)`` con correspondencia posicional 1:1.
    """
    entries = load_knowledge_base()
    texts = [e["embedding_text"] for e in entries]
    matrix = generate_embeddings(texts)
    ids = np.array([e["id"] for e in entries])

    out_path = CONFIG["hw_embeddings_path"]
    np.savez(out_path, embeddings=matrix, ids=ids)
    logging.info("Índice hardware guardado: %s (shape=%s)", out_path, matrix.shape)
    return matrix, entries


def build_ticket_index() -> tuple[np.ndarray, pd.DataFrame]:
    """Vectoriza el corpus de tickets y guarda el índice en disco.

    Returns:
        Tupla ``(matrix, corpus_df)`` con correspondencia posicional 1:1.
    """
    df = load_ticket_corpus()
    texts = df["search_text"].tolist()
    matrix = generate_embeddings(texts)
    ids = np.array(df["ticket_id"].tolist())

    out_path = CONFIG["ticket_embeddings_path"]
    np.savez(out_path, embeddings=matrix, ids=ids)
    logging.info("Índice tickets guardado: %s (shape=%s)", out_path, matrix.shape)
    return matrix, df


# ---------------------------------------------------------------------------
# CARGA O CONSTRUCCIÓN DE ÍNDICES
# ---------------------------------------------------------------------------

def _load_or_build_index(
    index_type: Literal["hardware", "tickets"],
) -> tuple[np.ndarray, Any]:
    """Carga el índice desde caché o lo reconstruye si no existe o se fuerza rebuild.

    Args:
        index_type: ``"hardware"`` o ``"tickets"``.

    Returns:
        Para hardware: ``(matrix, entries_list)``.
        Para tickets:  ``(matrix, corpus_df)``.
    """
    npz_path = (
        CONFIG["hw_embeddings_path"]
        if index_type == "hardware"
        else CONFIG["ticket_embeddings_path"]
    )

    if npz_path.exists() and not CONFIG["force_rebuild"]:
        logging.info("Índice cargado desde caché: %s", npz_path)
        data = np.load(npz_path, allow_pickle=False)
        matrix = data["embeddings"].astype(np.float32)
        if index_type == "hardware":
            return matrix, load_knowledge_base()
        else:
            return matrix, load_ticket_corpus()
    else:
        logging.info("Índice reconstruido: %s", npz_path)
        if index_type == "hardware":
            return build_hardware_index()
        else:
            return build_ticket_index()


# ---------------------------------------------------------------------------
# BÚSQUEDA SEMÁNTICA
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    index_type: Literal["hardware", "tickets"],
    top_k: int | None = None,
    only_closed: bool = False,
) -> list[dict]:
    """Búsqueda por similitud coseno entre la query y el índice indicado.

    Args:
        query: texto de la consulta.
        index_type: ``"hardware"`` o ``"tickets"``.
        top_k: número máximo de resultados; usa ``CONFIG["top_k"]`` si es None.
        only_closed: si True, filtra tickets abiertos (solo aplica a index_type="tickets").

    Returns:
        Lista de dicts ordenada por score descendente, filtrada por threshold.
        Cada dict tiene las claves: source, score, id, title, summary, metadata.
    """
    k = top_k if top_k is not None else CONFIG["top_k"]
    threshold = CONFIG["similarity_threshold"]

    query_vec = generate_embeddings([query], show_progress=False)[0]  # shape (dim,)
    matrix, meta = _load_or_build_index(index_type)

    scores = np.dot(matrix, query_vec)  # shape (n,)

    # Ordenar descendente y tomar top_k antes de filtrar por threshold
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < threshold:
            continue

        if index_type == "hardware":
            entry = meta[idx]
            result = {
                "source":   "hardware",
                "score":    round(score, 4),
                "id":       entry["id"],
                "title":    entry["problema"],
                "summary":  entry["embedding_text"],
                "metadata": {
                    k: v for k, v in entry.items()
                    if k not in {"id", "problema", "embedding_text"}
                },
            }
        else:
            row = meta.iloc[idx]
            result = {
                "source":   "tickets",
                "score":    round(score, 4),
                "id":       str(row["ticket_id"]),
                "title":    str(row["issue_type"]),
                "summary":  str(row["resolution_summary"]) if pd.notna(row["resolution_summary"]) else "",
                "metadata": {
                    "status":   str(row["status"]),
                    "is_open":  bool(row["is_open"]),
                },
            }
            if only_closed and index_type == "tickets" and bool(row["is_open"]):
                continue
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# BÚSQUEDA POR KEYWORDS
# ---------------------------------------------------------------------------

def keyword_search(query: str, top_k: int | None = None) -> list[dict]:
    """Búsqueda TF-IDF sobre el campo ``keywords`` de la Knowledge Base de hardware.

    Esta función **solo** busca dentro de la Knowledge Base de hardware.
    No realiza búsqueda sobre el corpus de tickets.

    Razón: Los tickets no poseen un campo ``keywords`` estructurado y curado
    como sí lo tienen las entradas de la Knowledge Base de hardware. Por este
    motivo, la búsqueda por palabras clave (TF-IDF) solo tiene sentido aplicarla
    sobre las entradas de hardware.

    Para buscar en tickets se debe utilizar:
        semantic_search(query, "tickets", top_k=top_k)

    Args:
        query: texto de la consulta.
        top_k: número máximo de resultados; usa ``CONFIG["top_k"]`` si es None.

    Returns:
        Lista de dicts en el mismo formato que ``semantic_search()``.
    """
    k = top_k if top_k is not None else CONFIG["top_k"]
    threshold = CONFIG["similarity_threshold"]
    t0 = time.perf_counter()

    vectorizer, tfidf_matrix, entries = _TfidfCache.get()
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < threshold:
            continue
        entry = entries[idx]
        results.append({
            "source":   "hardware",
            "score":    round(score, 4),
            "id":       entry["id"],
            "title":    entry["problema"],
            "summary":  entry["embedding_text"],
            "metadata": {
                kk: vv for kk, vv in entry.items()
                if kk not in {"id", "problema", "embedding_text"}
            },
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logging.info("keyword_search: %d resultados en %.1f ms", len(results), elapsed_ms)
    return results


# ---------------------------------------------------------------------------
# BÚSQUEDA HÍBRIDA
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int | None = None,
    alpha: float = 0.7,
) -> list[dict]:
    """Combina búsqueda semántica y keyword con peso configurable.

    Args:
        query: texto de la consulta.
        top_k: número máximo de resultados finales.
        alpha: peso semántico (0.0–1.0); el resto va a keyword score.

    Returns:
        Lista de dicts deduplicada por id, ordenada por score combinado descendente.
    """
    k = top_k if top_k is not None else CONFIG["top_k"]
    t0 = time.perf_counter()

    sem_results = semantic_search(query, "hardware", top_k=k)
    kw_results  = keyword_search(query, top_k=k)

    # Indexar por id para deduplicar y combinar scores
    combined: dict[str, dict] = {}

    for r in sem_results:
        combined[r["id"]] = {**r, "_sem_score": r["score"], "_kw_score": 0.0}

    for r in kw_results:
        if r["id"] in combined:
            combined[r["id"]]["_kw_score"] = r["score"]
        else:
            combined[r["id"]] = {**r, "_sem_score": 0.0, "_kw_score": r["score"]}

    results = []
    for item in combined.values():
        hybrid_score = alpha * item["_sem_score"] + (1 - alpha) * item["_kw_score"]
        results.append({
            "source":   item["source"],
            "score":    round(hybrid_score, 4),
            "id":       item["id"],
            "title":    item["title"],
            "summary":  item["summary"],
            "metadata": item["metadata"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logging.info("hybrid_search: %d resultados en %.1f ms (alpha=%.2f)", len(results[:k]), elapsed_ms, alpha)
    return results[:k]


# ---------------------------------------------------------------------------
# INTERFAZ PÚBLICA PARA FASE 3
# ---------------------------------------------------------------------------

def retrieve_relevant_knowledge(
    query: str,
    top_k: int | None = None,
) -> dict:
    """Punto de entrada único para Streamlit (Fase 3).

    Ejecuta búsqueda híbrida sobre hardware KB y búsqueda semántica sobre tickets.

    Args:
        query: pregunta o descripción del problema del usuario.
        top_k: número máximo de resultados por fuente.

    Returns:
        Dict con claves: query, hardware_results, ticket_results,
        total_results, search_time_ms.
    """
    t0 = time.perf_counter()

    hardware_results = hybrid_search(query, top_k=top_k)
    ticket_results   = semantic_search(query, "tickets", top_k=top_k)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.info(
        "retrieve_relevant_knowledge: hw=%d ticket=%d (%.1f ms)",
        len(hardware_results), len(ticket_results), elapsed_ms,
    )

    return {
        "query":            query,
        "hardware_results": hardware_results,
        "ticket_results":   ticket_results,
        "total_results":    len(hardware_results) + len(ticket_results),
        "search_time_ms":   round(elapsed_ms, 2),
    }


def get_similar_tickets(query: str, top_k: int = 3) -> list[dict]:
    """Retorna tickets similares a la query que ya están cerrados (resueltos).

    Args:
        query: texto de búsqueda.
        top_k: número de resultados a retornar.

    Returns:
        Subconjunto de ``semantic_search`` filtrado a tickets con is_open == False.
    """
    return semantic_search(query, "tickets", top_k=top_k, only_closed=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Motor NLP — Hatbot FAQ")
    parser.add_argument("--build-index", action="store_true",
                        help="Construye índices de embeddings para HW y tickets")
    parser.add_argument("--force", action="store_true",
                        help="Fuerza reconstrucción aunque los índices ya existan")
    parser.add_argument("--query",  type=str, default=None,
                        help="Consulta en lenguaje natural")
    parser.add_argument("--top-k",  type=int, default=CONFIG["top_k"],
                        help="Número máximo de resultados")
    args = parser.parse_args()

    if args.force:
        CONFIG["force_rebuild"] = True

    if args.build_index:
        logging.info("Construyendo índice hardware...")
        build_hardware_index()
        logging.info("Construyendo índice tickets...")
        build_ticket_index()
        logging.info("Índices listos.")

    if args.query:
        result = retrieve_relevant_knowledge(args.query, top_k=args.top_k)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(json.dumps(result, ensure_ascii=False, indent=2))
