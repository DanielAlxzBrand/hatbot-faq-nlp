"""
scripts/verificar_kb.py
=======================
Paso 5: verificación post-generalización de hardware_knowledge_base.json.

Comprobaciones:
  1. JSON válido y con la clave "hardware_entries".
  2. Ningún campo contiene cadenas de marcas/modelos específicos de equipo
     (PATRON_ESPECIFICO).
  3. Todos los embedding_text tienen al menos MIN_EMBED_CHARS caracteres.
  4. Informe de longitud real de cada embedding_text.
  5. Informe de keywords y referencias de cada entrada.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_processor import PATHS  # noqa: E402

KB_PATH = Path(PATHS["hw_kb"])

# ---------------------------------------------------------------------------
# Patrón de términos específicos que NO deben aparecer en ningún campo de texto
# (excepto en campos que no son strings planos, como listas de herramientas).
# ---------------------------------------------------------------------------
PATRON_ESPECIFICO = re.compile(
    r"\b(Acer Aspire|Aspire Lite|Asus X441|X441UV|Janus|N24H3)\b",
    re.IGNORECASE,
)

MIN_EMBED_CHARS = 800   # mínimo exigido en embedding_text

# Campos de texto libre donde buscar el patrón
CAMPOS_TEXTO = [
    "modelo_equipo",
    "problema",
    "embedding_text",
]
# Campos que son listas de strings
CAMPOS_LISTAS = [
    "keywords",
    "referencias",
    "sintomas",
    "pasos_diagnostico",
    "pasos_solucion",
    "prevencion",
]

# ---------------------------------------------------------------------------

def check_pattern(text: str, campo: str, entry_id: str) -> list[str]:
    """Retorna lista de errores si el patrón se encuentra en `text`."""
    errores = []
    for m in PATRON_ESPECIFICO.finditer(text):
        errores.append(
            f"  [FAIL] {entry_id}.{campo}: término específico encontrado → «{m.group()}» "
            f"en pos {m.start()}"
        )
    return errores


def main() -> None:
    print("=" * 72)
    print("PASO 5 — VERIFICACIÓN POST-GENERALIZACIÓN")
    print("=" * 72)

    # ── 1. Cargar JSON ───────────────────────────────────────────────────
    if not KB_PATH.exists():
        print(f"[FAIL] No se encontró el archivo: {KB_PATH.resolve()}")
        sys.exit(1)

    try:
        with open(KB_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[FAIL] JSON inválido: {exc}")
        sys.exit(1)

    entries = data.get("hardware_entries")
    if not isinstance(entries, list) or not entries:
        print("[FAIL] La clave 'hardware_entries' está ausente o vacía.")
        sys.exit(1)

    print(f"\n  [OK] JSON válido — {len(entries)} entradas cargadas.\n")

    total_errores    = 0
    embed_advertencias = 0

    for entry in entries:
        eid = entry.get("id", "?")
        print(f"  {'─'*66}")
        print(f"  {eid}")
        errores_entry = []

        # ── 2. Búsqueda de patrón en campos de texto plano ────────────
        for campo in CAMPOS_TEXTO:
            val = entry.get(campo, "")
            if isinstance(val, str):
                errores_entry.extend(check_pattern(val, campo, eid))

        # ── 3. Búsqueda de patrón en campos que son listas ─────────────
        for campo in CAMPOS_LISTAS:
            items = entry.get(campo, [])
            for item in items:
                if isinstance(item, str):
                    errores_entry.extend(check_pattern(item, campo, eid))

        # ── 4. Longitud de embedding_text ──────────────────────────────
        et = entry.get("embedding_text", "")
        et_len = len(et)
        if et_len < MIN_EMBED_CHARS:
            errores_entry.append(
                f"  [FAIL] {eid}.embedding_text: {et_len} chars < {MIN_EMBED_CHARS} mínimos"
            )
            embed_advertencias += 1
        else:
            print(f"    embedding_text : {et_len} chars  ✓")

        # ── 5. Mostrar keywords y referencias ──────────────────────────
        kws  = entry.get("keywords", [])
        refs = entry.get("referencias", [])
        modelo = entry.get("modelo_equipo", "—")
        print(f"    modelo_equipo  : {modelo!r}")
        print(f"    keywords ({len(kws):2d})  : {kws}")
        print(f"    referencias    :")
        for r in refs:
            print(f"      - {r}")

        # ── Reportar errores de esta entrada ──────────────────────────
        if errores_entry:
            for e in errores_entry:
                print(e)
            total_errores += len(errores_entry)
        else:
            print(f"    patron_check   : LIMPIO  ✓")

    # ── Resumen final ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    if total_errores == 0:
        print("  [RESULTADO] TODOS LOS CONTROLES PASARON CORRECTAMENTE")
        print(f"  - Entradas verificadas      : {len(entries)}")
        print(f"  - Errores de patrón         : 0")
        print(f"  - embedding_text insuficientes: {embed_advertencias}")
    else:
        print(f"  [RESULTADO] {total_errores} ERROR(ES) ENCONTRADO(S)")
        print(f"  Revisa los campos marcados con [FAIL] arriba.")
        sys.exit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
