"""
Fase 1 — Hatbot FAQ NLP: carga, limpieza y base de conocimiento de hardware.
"""

import json
import logging
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. CONSTANTES GLOBALES
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"

PATHS = {
    "raw_csv":    Path("data/raw/synthetic_it_support_tickets.csv"),
    "clean_parq": Path("data/processed/tickets_clean.parquet"),
    "hw_kb":      Path("data/custom/hardware_knowledge_base.json"),
}

EXPECTED_COLUMNS = [
    "ticket_id", "created_at", "customer_id", "customer_segment", "channel",
    "product_area", "issue_type", "priority", "status", "sla_plan",
    "initial_message", "agent_first_reply", "resolution_summary",
    "resolution_time_hours", "reopened", "customer_sentiment", "csat_score",
    "has_attachment", "platform", "region",
]

# created_at va en parse_dates; reopened y has_attachment como bool desde int
TICKET_DTYPES = {
    "ticket_id":            pd.StringDtype(),
    "customer_id":          pd.StringDtype(),
    "customer_segment":     "category",
    "channel":              "category",
    "product_area":         "category",
    "issue_type":           "category",
    "priority":             "category",
    "status":               "category",
    "sla_plan":             "category",
    "initial_message":      pd.StringDtype(),
    "agent_first_reply":    pd.StringDtype(),
    "resolution_summary":   pd.StringDtype(),
    "resolution_time_hours": pd.Float32Dtype(),
    "reopened":             pd.BooleanDtype(),
    "customer_sentiment":   "category",
    "csat_score":           pd.Int16Dtype(),
    "has_attachment":       pd.BooleanDtype(),
    "platform":             "category",
    "region":               "category",
}

_CATEGORY_COLS = [
    col for col, dtype in TICKET_DTYPES.items() if dtype == "category"
]

_CLOSED_STATUSES = {"resolved", "closed_no_action"}


# ---------------------------------------------------------------------------
# 2. FUNCIONES
# ---------------------------------------------------------------------------

def load_and_clean_tickets(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV, valida el esquema, normaliza y enriquece el DataFrame."""
    df = pd.read_csv(
        csv_path,
        dtype=TICKET_DTYPES,
        parse_dates=["created_at"],
        low_memory=False,
    )

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas ausentes en el CSV: {missing}. "
            "Verifica que el archivo fuente no haya sido modificado."
        )

    # Strip + lowercase en columnas category
    for col in _CATEGORY_COLS:
        df[col] = df[col].cat.rename_categories(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    # is_open: True cuando el ticket NO está en un estado cerrado
    df["is_open"] = (~df["status"].isin(_CLOSED_STATUSES)).astype(pd.BooleanDtype())

    # resolution_time_hours → NA en tickets abiertos (vectorizado)
    df["resolution_time_hours"] = df["resolution_time_hours"].where(~df["is_open"], pd.NA)

    df["schema_version"] = SCHEMA_VERSION

    return df


def save_clean_data(df: pd.DataFrame, output_path: Path) -> dict:
    """Guarda el DataFrame limpio en Parquet (fastparquet/snappy) y retorna métricas."""
    mb_before = df.memory_usage(deep=True).sum() / 1_048_576

    df.to_parquet(output_path, engine="fastparquet", compression="snappy", index=False)

    mb_after = output_path.stat().st_size / 1_048_576
    ratio = mb_before / mb_after if mb_after > 0 else float("inf")

    return {
        "rows":      int(len(df)),
        "columns":   int(len(df.columns)),
        "mb_before": round(mb_before, 3),
        "mb_after":  round(mb_after, 3),
        "ratio":     round(ratio, 2),
    }


def get_dataframe_stats(df: pd.DataFrame) -> dict:
    """Retorna estadísticas descriptivas serializables a JSON."""
    open_mask   = df["is_open"] == True
    closed_mask = df["is_open"] == False

    avg_res = df.loc[closed_mask, "resolution_time_hours"].mean()
    avg_res = float(round(avg_res, 2)) if not pd.isna(avg_res) else None

    date_min = df["created_at"].min()
    date_max = df["created_at"].max()

    return {
        "total_tickets":         int(len(df)),
        "open_tickets":          int(open_mask.sum()),
        "closed_tickets":        int(closed_mask.sum()),
        "categories":            df["issue_type"].value_counts().to_dict(),
        "avg_resolution_hours":  avg_res,
        "null_counts":           {c: int(df[c].isna().sum()) for c in df.columns},
        "memory_mb":             round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        "date_range": {
            "min": str(date_min.date()) if not pd.isna(date_min) else None,
            "max": str(date_max.date()) if not pd.isna(date_max) else None,
        },
    }


def create_hardware_knowledge_base() -> dict:
    """Carga la Knowledge Base de hardware desde disco o genera la base inicial.

    Comportamiento (idempotente):
    - Si ``PATHS["hw_kb"]`` existe en disco: carga y retorna su contenido sin
      modificarlo. Preserva las entradas enriquecidas externamente (HW_007–HW_015).
    - Si ``PATHS["hw_kb"]`` no existe: genera las 6 entradas iniciales como
      bootstrap. Este es el único caso en que se hardcodean entradas.

    Returns:
        Dict con claves ``schema_version`` y ``hardware_entries``.
    """
    hw_path = PATHS["hw_kb"]

    if hw_path.exists():
        logger.info(
            "Knowledge Base existente encontrada en %s — cargando sin regenerar.",
            hw_path.resolve(),
        )
        with open(hw_path, encoding="utf-8") as fh:
            data = json.load(fh)
        n = len(data.get("hardware_entries", []))
        logger.info("Knowledge Base cargada desde disco: %d entradas.", n)
        return data

    logger.warning(
        "Knowledge Base no encontrada en %s — generando 6 entradas iniciales (bootstrap).",
        hw_path.resolve(),
    )
    return _create_initial_knowledge_base()


def _create_initial_knowledge_base() -> dict:
    """Genera las 6 entradas iniciales de la Knowledge Base (solo para bootstrap).

    Esta función solo se ejecuta cuando ``data/custom/hardware_knowledge_base.json``
    no existe. Para actualizar o ampliar la KB, editar el JSON directamente —
    nunca modificar esta función con fines de actualización.

    Returns:
        Dict con claves ``schema_version`` y ``hardware_entries`` (6 entradas).
    """
    entries = [
        {
            "id": "HW_001",
            "categoria": "Mantenimiento preventivo",
            "modelo_equipo": "Acer Aspire Lite 16-51P",
            "problema": "Sobrecalentamiento por acumulación de polvo y pasta térmica degradada",
            "sintomas": [
                "Temperatura CPU >90 °C en carga moderada",
                "Throttling térmico: bajada brusca de frecuencia",
                "Ventilador a máximas RPM de forma constante",
                "Apagados inesperados por protección térmica",
            ],
            "keywords": [
                "sobrecalentamiento", "polvo", "pasta térmica", "Arctic MX-4",
                "Arctic MX-6", "soplador eléctrico", "brocha antiestática",
                "Acer Aspire", "throttling",
            ],
            "herramientas_necesarias": [
                "Destornillador Philips PH0 y PH1",
                "Soplador eléctrico (no lata de aire comprimido)",
                "Brocha antiestática de cerdas suaves",
                "Pasta térmica Arctic MX-4 o MX-6",
                "Isopropanol ≥90 % + hisopos sin pelusa",
                "Pulsera antiestática",
            ],
            "pasos_diagnostico": [
                "Instalar HWMonitor Pro y registrar temperaturas en reposo y bajo carga (Cinebench R23).",
                "Verificar velocidad del ventilador; RPM constante máxima indica obstrucción.",
                "Revisar rejillas de ventilación con linterna; polvo visible confirma diagnóstico.",
            ],
            "pasos_solucion": [
                "Apagar, desconectar corriente y batería (si es extraíble).",
                "Retirar tornillos de la tapa inferior siguiendo la secuencia marcada en el chasis.",
                "Desconectar el conector de la batería interna antes de tocar cualquier componente.",
                "Usar soplador eléctrico en ráfagas cortas sobre ventilador y disipador; evacuar polvo hacia el exterior.",
                "Limpiar aspas del ventilador con brocha antiestática.",
                "Retirar tornillos del disipador en orden cruzado para liberar presión uniforme.",
                "Limpiar restos de pasta vieja con isopropanol ≥90 % e hisopos sin pelusa.",
                "Aplicar gota de pasta Arctic MX-4/MX-6 del tamaño de un guisante sobre el die de la CPU.",
                "Reinstalar disipador en orden cruzado; ajustar sin sobrepasar el torque recomendado.",
                "Reconectar batería, cerrar tapa y verificar temperaturas nuevamente.",
            ],
            "tiempo_estimado_minutos": 45,
            "nivel_dificultad": "Intermedio",
            "prevencion": [
                "Repetir limpieza cada 12 meses en entornos con polvo moderado.",
                "No usar el equipo sobre superficies blandas que bloqueen la ventilación.",
                "Cambiar pasta térmica cada 2-3 años o cuando la temperatura en reposo supere 55 °C.",
            ],
            "referencias": [
                "Hoja de datos Arctic MX-4 / MX-6",
                "Manual de servicio Acer Aspire Lite 16-51P (N24H3)",
            ],
            "embedding_text": (
                "Sobrecalentamiento Acer Aspire Lite 16-51P por polvo y pasta térmica degradada. "
                "Síntomas: temperatura CPU mayor a 90 grados, throttling, ventilador a máximas RPM, apagados. "
                "Keywords: sobrecalentamiento polvo pasta térmica Arctic MX-4 MX-6 soplador brocha antiestática throttling. "
                "Solución: limpiar con soplador eléctrico y brocha, reemplazar pasta térmica Arctic MX-4 o MX-6, "
                "retirar disipador en orden cruzado, aplicar pasta nueva."
            ),
        },
        {
            "id": "HW_002",
            "categoria": "Mantenimiento preventivo",
            "modelo_equipo": "Asus X441UV",
            "problema": "Degradación térmica y polvo en disipador con riesgo de daño en flex cables",
            "sintomas": [
                "Calor excesivo en zona del teclado y palm rest",
                "Errores de GPU discreta en carga gráfica",
                "Ruido inusual al abrir o cerrar la pantalla (flex cable tenso)",
                "Ventilador audible incluso en reposo",
            ],
            "keywords": [
                "Asus X441UV", "flex cable", "disipador chipset", "pasta térmica",
                "tornillos secuencia", "mantenimiento preventivo", "GPU", "calor",
            ],
            "herramientas_necesarias": [
                "Destornillador Philips PH0",
                "Spudger de plástico para palanca",
                "Soplador eléctrico",
                "Pasta térmica",
                "Isopropanol ≥90 %",
                "Pinzas antiestáticas",
            ],
            "pasos_diagnostico": [
                "Ejecutar GPU-Z para verificar temperatura de GPU discreta bajo carga.",
                "Palpar zona de salida de aire; ausencia de flujo indica bloqueo.",
                "Inspeccionar bisagras; resistencia excesiva puede dañar el flex de pantalla.",
            ],
            "pasos_solucion": [
                "Apagar y desconectar la alimentación.",
                "Retirar tornillos de la tapa inferior siguiendo la secuencia numérica serigrafada.",
                "Desconectar batería antes de proseguir.",
                "Liberar flex cables de pantalla y WiFi con palanca de plástico; no tirar.",
                "Retirar tornillos del disipador de CPU y chipset en orden cruzado.",
                "Limpiar polvo del disipador y ventilador con soplador.",
                "Aplicar pasta térmica fresca en CPU y chipset discretamente.",
                "Reinstalar disipador; verificar que los flex cables queden sin tensión.",
                "Cerrar tapa y confirmar que los flex no quedan pinzados.",
                "Encender y verificar temperaturas con GPU-Z y HWMonitor.",
            ],
            "tiempo_estimado_minutos": 60,
            "nivel_dificultad": "Intermedio",
            "prevencion": [
                "Limpiar cada 12 meses.",
                "Inspeccionar flex cables al abrir para detectar desgaste temprano.",
                "Evitar doblar la pantalla más allá del ángulo máximo diseñado.",
            ],
            "referencias": [
                "Manual de servicio Asus X441UV",
                "Foro de soporte Asus — sección mantenimiento X441",
            ],
            "embedding_text": (
                "Degradación térmica Asus X441UV con riesgo en flex cables y disipador de chipset. "
                "Síntomas: calor excesivo, errores GPU, ruido en bisagra, ventilador en reposo. "
                "Keywords: Asus X441UV flex cable disipador chipset pasta térmica tornillos secuencia GPU calor. "
                "Solución: retirar tornillos en secuencia, liberar flex con spudger, limpiar disipador, "
                "aplicar pasta térmica en CPU y chipset, reinstalar sin tensar flex cables."
            ),
        },
        {
            "id": "HW_003",
            "categoria": "Diagnóstico energético",
            "modelo_equipo": "Monitores Janus",
            "problema": "Fallo en fuente de alimentación interna con capacitores hinchados",
            "sintomas": [
                "Monitor no enciende o parpadea al arrancar",
                "Imagen con franjas horizontales o pérdida de brillo progresiva",
                "Sonido de clic repetitivo al intentar encender",
                "Olor a quemado cerca de la fuente interna",
            ],
            "keywords": [
                "Janus", "monitor", "capacitor hinchado", "ESR", "fuente interna",
                "altavoces integrados", "parpadeo", "no enciende", "diagnóstico energía",
            ],
            "herramientas_necesarias": [
                "Multímetro digital",
                "Medidor ESR de condensadores",
                "Cautín de temperatura regulable",
                "Soldadura de estaño sin plomo",
                "Capacitores de repuesto (misma capacidad y voltaje o superior)",
                "Destornillador Torx T8",
            ],
            "pasos_diagnostico": [
                "Medir tensión en conector de salida de la fuente; ausencia de tensión confirma fallo.",
                "Inspeccionar visualmente la placa de la fuente buscando capacitores con tapa abombada o electrolito derramado.",
                "Medir ESR de cada capacitor de la etapa primaria; valor >5 Ω indica degradación.",
                "Verificar fusible de línea con multímetro en modo continuidad.",
                "Comprobar altavoces integrados en modo audio para aislar el fallo a la fuente.",
            ],
            "pasos_solucion": [
                "Desconectar el monitor de la red eléctrica y esperar 5 minutos para descarga de capacitores.",
                "Abrir la carcasa con destornillador Torx T8; separar panel con espátula de plástico.",
                "Localizar y desoldear los capacitores hinchados con cautín.",
                "Soldar capacitores nuevos respetando la polaridad (banda blanca = negativo).",
                "Verificar fusible; reemplazar si está abierto.",
                "Cerrar carcasa y realizar prueba de encendido con carga mínima.",
                "Comprobar imagen, brillo y audio con señal de prueba.",
            ],
            "tiempo_estimado_minutos": 90,
            "nivel_dificultad": "Avanzado",
            "prevencion": [
                "Usar protector de sobretensión (UPS o supresor de picos).",
                "Evitar ambientes con humedad >60 % que acelera la degradación de electrolitos.",
                "Inspeccionar la placa anualmente si el monitor tiene más de 5 años.",
            ],
            "referencias": [
                "Hoja de datos capacitores Nichicon serie HE",
                "Norma IPC-7711/7721 para reparación de placas electrónicas",
            ],
            "embedding_text": (
                "Fallo fuente interna monitores Janus por capacitores hinchados ESR elevado. "
                "Síntomas: no enciende, parpadeo, franjas horizontales, clic repetitivo, olor quemado. "
                "Keywords: Janus monitor capacitor hinchado ESR fuente interna altavoces parpadeo diagnóstico energía. "
                "Solución: medir ESR, desoldear capacitores dañados, soldar repuestos con polaridad correcta, "
                "verificar fusible, comprobar imagen y audio."
            ),
        },
        {
            "id": "HW_004",
            "categoria": "Firmware y BIOS",
            "modelo_equipo": "PC genérico (placas ATX/mATX)",
            "problema": "Fallo de POST por configuración BIOS corrupta o pila CR2032 agotada",
            "sintomas": [
                "PC no pasa del POST; pantalla en negro o código de error en display de placa",
                "Fecha y hora se resetean a 2000-01-01 en cada arranque",
                "Beep codes de error de memoria o CPU",
                "BIOS no recuerda configuración de boot",
            ],
            "keywords": [
                "BIOS", "POST", "pila CR2032", "CLR_CMOS", "jumper", "reset BIOS",
                "flasheo USB", "no arranca", "beep code", "configuración corrupta",
            ],
            "herramientas_necesarias": [
                "Pila CR2032 nueva",
                "USB ≥8 GB formateado FAT32",
                "Herramienta oficial del fabricante de placa (ASUS EZ Flash, MSI M-Flash, Gigabyte Q-Flash)",
                "Multímetro (para verificar tensión de pila: debe ser ≥3.0 V)",
            ],
            "pasos_diagnostico": [
                "Anotar el código de error en el display Q-Code o interpretar beep codes según el manual.",
                "Medir tensión de la pila CR2032 con multímetro; <3.0 V indica reemplazo.",
                "Verificar si el problema persiste tras retirar módulos RAM excepto uno en slot A2.",
                "Intentar arranque con configuración mínima: CPU + 1 RAM + GPU integrada.",
            ],
            "pasos_solucion": [
                "Apagar y desconectar el cable de corriente.",
                "Retirar la pila CR2032 con espátula de plástico; esperar 30 segundos.",
                "Localizar el jumper CLR_CMOS (generalmente cerca de la pila) y puentear pines 1-2 durante 10 segundos.",
                "Volver el jumper a posición original e insertar pila CR2032 nueva.",
                "Si el problema persiste, descargar la BIOS más reciente del sitio del fabricante.",
                "Copiar el archivo de BIOS a raíz del USB FAT32 (sin renombrar).",
                "Iniciar el flasheo desde la utilidad de la placa (EZ Flash / M-Flash / Q-Flash) según fabricante.",
                "No interrumpir el proceso de flasheo; esperar al reinicio automático.",
                "Configurar fecha/hora y secuencia de boot en la nueva BIOS.",
            ],
            "tiempo_estimado_minutos": 30,
            "nivel_dificultad": "Intermedio",
            "prevencion": [
                "Reemplazar la pila CR2032 cada 5 años de forma preventiva.",
                "Exportar perfil de BIOS a USB antes de actualizaciones.",
                "Mantener la BIOS actualizada para compatibilidad con nuevos procesadores.",
            ],
            "referencias": [
                "Manual de placa base del fabricante (sección CLR_CMOS y BIOS Update)",
                "Base de datos de beep codes AMI/Award/Phoenix BIOS",
            ],
            "embedding_text": (
                "Fallo POST en PC genérico por BIOS corrupta o pila CR2032 agotada. "
                "Síntomas: no arranca, pantalla negra, beep codes, fecha reseteada, BIOS sin memoria. "
                "Keywords: BIOS POST pila CR2032 CLR_CMOS jumper reset flasheo USB no arranca beep code. "
                "Solución: medir pila, puentear CLR_CMOS, reemplazar CR2032, flashear BIOS desde USB FAT32 "
                "con herramienta del fabricante."
            ),
        },
        {
            "id": "HW_005",
            "categoria": "Diagnóstico térmico",
            "modelo_equipo": "PC genérico (escritorio y portátil)",
            "problema": "Throttling térmico detectado con HWMonitor Pro y Cinebench R23",
            "sintomas": [
                "Rendimiento inconsistente: velocidad varía sin carga aparente",
                "Frecuencia de CPU/GPU cae por debajo de la base clock",
                "Temperaturas >95 °C en CPU bajo carga sostenida",
                "Puntuación Cinebench Multi muy por debajo del valor de referencia",
            ],
            "keywords": [
                "throttling", "HWMonitor Pro", "Cinebench R23", "temperatura CPU",
                "frecuencia base", "rendimiento", "diagnóstico térmico", "TjMax",
            ],
            "herramientas_necesarias": [
                "HWMonitor Pro (CPUID)",
                "Cinebench R23",
                "ThrottleStop (portátiles Intel) o AMD Ryzen Master",
                "HWiNFO64 para log detallado",
            ],
            "pasos_diagnostico": [
                "Instalar HWMonitor Pro y HWiNFO64; iniciar logging.",
                "Ejecutar Cinebench R23 Multi Core durante 10 minutos.",
                "Monitorear: temperatura máxima CPU, Package Power (W), frecuencia real vs nominal.",
                "Si la frecuencia cae >20 % por debajo del boost clock mientras la temperatura supera 90 °C, hay throttling térmico.",
                "Si la frecuencia cae pero la temperatura es normal (<75 °C), el throttling es por límite de potencia (PL1/PL2).",
                "Revisar el log de HWiNFO64 para identificar el tipo de throttling: Thermal, Power, Current.",
            ],
            "pasos_solucion": [
                "Throttling térmico: limpiar sistema de refrigeración y reemplazar pasta térmica (ver HW_001).",
                "Throttling por potencia en portátiles: usar ThrottleStop para ajustar PL1/PL2 dentro de los límites del fabricante.",
                "Throttling por corriente: verificar que el adaptador de corriente tenga la potencia correcta.",
                "En escritorio: revisar que el cooler de CPU esté correctamente montado y el flujo de aire del gabinete sea adecuado.",
                "Documentar las temperaturas antes y después de la intervención para comparar.",
            ],
            "tiempo_estimado_minutos": 40,
            "nivel_dificultad": "Intermedio",
            "prevencion": [
                "Ejecutar Cinebench mensualmente para detectar degradación térmica temprana.",
                "Mantener el sistema de refrigeración limpio y la pasta térmica vigente.",
                "Asegurar flujo de aire positivo en el gabinete (más entrada que salida).",
            ],
            "referencias": [
                "Documentación CPUID HWMonitor Pro",
                "Intel ARK — TjMax por generación de CPU",
                "Maxon Cinebench R23 Hardware Database",
            ],
            "embedding_text": (
                "Throttling térmico en PC genérico diagnosticado con HWMonitor Pro y Cinebench R23. "
                "Síntomas: frecuencia CPU cae bajo carga, temperaturas mayores a 95 grados, rendimiento inconsistente. "
                "Keywords: throttling HWMonitor Pro Cinebench R23 temperatura CPU frecuencia base TjMax rendimiento. "
                "Solución: identificar tipo de throttling térmico o por potencia, limpiar refrigeración, "
                "ajustar PL1 PL2 con ThrottleStop, verificar adaptador de corriente."
            ),
        },
        {
            "id": "HW_006",
            "categoria": "Herramientas de diagnóstico USB",
            "modelo_equipo": "PC genérico",
            "problema": "Creación de USB bootable multiherramienta para diagnóstico de hardware",
            "sintomas": [
                "Sospecha de fallo de RAM (pantallazos azules, errores aleatorios)",
                "Disco duro con sectores defectuosos o velocidad de lectura degradada",
                "Sistema no arranca y se requiere diagnóstico previo a reinstalación",
                "Necesidad de verificar salud del disco antes de garantía o RMA",
            ],
            "keywords": [
                "Ventoy", "MemTest86", "Victoria HDD", "CrystalDiskInfo", "MHDD",
                "USB bootable", "diagnóstico RAM", "diagnóstico disco", "sectores defectuosos",
                "S.M.A.R.T.", "RMA",
            ],
            "herramientas_necesarias": [
                "USB ≥8 GB (preferible ≥16 GB para múltiples ISOs)",
                "Ventoy (ventoy.net) — gestor multiboot",
                "MemTest86 (memtest86.com) — ISO gratuita",
                "Victoria HDD (versión para DOS/UEFI)",
                "CrystalDiskInfo (para Windows; revisar S.M.A.R.T. antes de bootear)",
                "MHDD (diagnóstico avanzado de disco en modo DOS)",
            ],
            "pasos_diagnostico": [
                "En Windows: ejecutar CrystalDiskInfo para lectura previa de S.M.A.R.T. y detectar sectores reubicados.",
                "Identificar qué componente sospechar: RAM (errores aleatorios) o disco (lentitud, errores de lectura).",
                "Preparar el USB Ventoy como paso previo a cualquier diagnóstico booteable.",
            ],
            "pasos_solucion": [
                "Instalar Ventoy en el USB: ejecutar Ventoy2Disk.exe, seleccionar el USB y hacer clic en Install.",
                "Copiar las ISOs descargadas (MemTest86, Victoria, MHDD) a la partición Ventoy sin descomprimir.",
                "Reiniciar el equipo y arrancar desde el USB (tecla de boot: F8/F11/F12 según fabricante).",
                "Seleccionar MemTest86 en el menú Ventoy; ejecutar al menos 2 pasadas completas (≈40 min por pasada).",
                "Si MemTest86 muestra errores: la RAM está dañada; aislar módulos para identificar el defectuoso.",
                "Seleccionar Victoria HDD para diagnóstico de disco: ejecutar scan de superficie y observar bloques marcados en rojo (B).",
                "Usar MHDD para diagnóstico avanzado si Victoria indica sectores dudosos; documentar LBA de los sectores.",
                "Comparar resultados con CrystalDiskInfo S.M.A.R.T. para decidir reemplazo o RMA.",
            ],
            "tiempo_estimado_minutos": 120,
            "nivel_dificultad": "Básico",
            "prevencion": [
                "Mantener el USB de diagnóstico actualizado con las últimas versiones de las ISOs.",
                "Revisar S.M.A.R.T. mensualmente con CrystalDiskInfo en discos >3 años.",
                "Hacer backup antes de cualquier diagnóstico invasivo de disco.",
            ],
            "referencias": [
                "Ventoy Documentation — ventoy.net/en/doc_start.html",
                "MemTest86 User Guide — memtest86.com/technical.htm",
                "Victoria HDD Manual — hddscan.com/doc/victoria.pdf",
                "CrystalDiskInfo — crystalmark.info/en/software/crystaldiskinfo",
            ],
            "embedding_text": (
                "Creación USB bootable multiherramienta con Ventoy para diagnóstico de hardware en PC genérico. "
                "Síntomas: pantallazos azules por RAM, sectores defectuosos en disco, sistema que no arranca. "
                "Keywords: Ventoy MemTest86 Victoria HDD CrystalDiskInfo MHDD USB bootable diagnóstico RAM disco SMART RMA. "
                "Solución: instalar Ventoy en USB, copiar ISOs, ejecutar MemTest86 dos pasadas, "
                "escanear disco con Victoria HDD, documentar sectores y comparar con SMART para decidir RMA."
            ),
        },
    ]

    return {
        "schema_version":   SCHEMA_VERSION,
        "hardware_entries": entries,
    }


def save_knowledge_base(kb: dict, output_path: Path) -> None:
    """Serializa la base de conocimiento a JSON con formato legible."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(kb, fh, indent=2, ensure_ascii=False)
    n = len(kb.get("hardware_entries", []))
    logger.info("Knowledge base guardada: %d entradas → %s", n, output_path.resolve())


# ---------------------------------------------------------------------------
# 3. BLOQUE PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # [1/5] Leer CSV
    logger.info("[1/5] Leyendo CSV: %s", PATHS["raw_csv"])
    raw_path = PATHS["raw_csv"]

    # [2/5] Limpiar y validar
    logger.info("[2/5] Limpiando y validando datos...")
    df = load_and_clean_tickets(raw_path)
    logger.info("      %d filas × %d columnas cargadas.", len(df), len(df.columns))

    # [3/5] Guardar Parquet
    logger.info("[3/5] Guardando Parquet: %s", PATHS["clean_parq"])
    metrics = save_clean_data(df, PATHS["clean_parq"])
    logger.info(
        "      Guardado — filas: %d | columnas: %d | antes: %.3f MB | después: %.3f MB | ratio: %.2fx",
        metrics["rows"], metrics["columns"],
        metrics["mb_before"], metrics["mb_after"], metrics["ratio"],
    )

    # [4/5] Cargar o generar knowledge base
    logger.info("[4/5] Cargando o generando base de conocimiento de hardware...")
    kb = create_hardware_knowledge_base()
    logger.info(
        "      %d entradas disponibles (desde disco si existía, generadas si no).",
        len(kb["hardware_entries"]),
    )

    # [5/5] Guardar JSON
    logger.info("[5/5] Guardando JSON: %s", PATHS["hw_kb"])
    save_knowledge_base(kb, PATHS["hw_kb"])

    # Estadísticas finales
    stats = get_dataframe_stats(df)
    logger.info("Estadísticas finales:\n%s", pformat(stats))
