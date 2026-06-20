"""
scripts/generalizar_kb.py
=========================
TAREA: Generalizar entradas HW_001-HW_006 de la Knowledge Base de hardware.

Qué hace este script
--------------------
1. Lee hardware_knowledge_base.json desde PATHS["hw_kb"].
2. Crea un backup automático con timestamp antes de tocar el archivo.
3. Aplica el diccionario `CAMBIOS` sobre cada entrada afectada:
   - Elimina referencias a marcas/modelos de equipo (Acer Aspire, Asus X441UV, Janus).
   - Generaliza modelo_equipo, keywords y referencias.
   - Reemplaza embedding_text con prosa técnica densa (800-2000 caracteres)
     extraída del PDF "Embeddings_HW_001_a_HW_015.pdf.pdf".
4. Guarda el JSON modificado con indent=2, ensure_ascii=False.
5. Imprime un resumen campo a campo de cada cambio aplicado.

Ejecutar:
    python scripts/generalizar_kb.py

Prerequisito: el JSON ya debe existir en PATHS["hw_kb"].
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Forzar stdout en UTF-8 para que los caracteres especiales del plan no fallen en Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_processor import PATHS  # noqa: E402

KB_PATH    = Path(PATHS["hw_kb"])
BACKUP_DIR = KB_PATH.parent / "backups"

# ---------------------------------------------------------------------------
# PASO 1C — Plan de cambios campo a campo
# ---------------------------------------------------------------------------
PLAN_TEXT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           PASO 1C — PLAN DE CAMBIOS CAMPO A CAMPO (HW_001–HW_006)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

HW_001  (Sobrecalentamiento — pasta térmica y polvo)
  ● modelo_equipo : "Acer Aspire Lite 16-51P"  →  "Laptops en general"
  ● keywords      : eliminar "Acer Aspire"; añadir "Pump-out Effect", "TjMax",
                    "HWiNFO64", "Cinebench R23", "Thermal Grizzly Kryonaut"
  ● referencias   : eliminar "Manual de servicio Acer Aspire Lite 16-51P (N24H3)"
                    añadir "Hoja de datos Thermal Grizzly Kryonaut"
                    añadir "Documentación HWiNFO64 — sensores y banderas de throttling"
  ● embedding_text: 449 chars → ~1750 chars (fuente: PDF págs. 1-3)

HW_002  (Mantenimiento con GPU dedicada — flex cables)
  ● modelo_equipo : "Asus X441UV"  →  "Laptops con GPU dedicada"
  ● keywords      : eliminar "Asus X441UV"; añadir "almohadilla térmica",
                    "Pump-out Effect", "desarme laptop", "HWiNFO64"
  ● pasos_solucion[1] : "...siguiendo la secuencia numérica serigrafada"
                    →  "...siguiendo la secuencia indicada en el chasis
                        o en el manual de servicio del modelo"
  ● referencias   : eliminar "Manual de servicio Asus X441UV"
                    eliminar "Foro de soporte Asus — sección mantenimiento X441"
                    añadir "Hoja de datos Arctic MX-4 / MX-6"
                    añadir "Hoja de datos Thermal Grizzly Kryonaut"
                    añadir "Guía general de mantenimiento de laptops con GPU dedicada"
  ● embedding_text: 318 chars → ~1850 chars (fuente: PDF págs. 4-5)

HW_003  (Fuente de alimentación — capacitores en monitores)
  ● modelo_equipo : "Monitores Janus"  →  "Monitores LCD/LED"
  ● keywords      : eliminar "Janus"; añadir "MOSFET", "driver backlight",
                    "soldadura fría", "fuente conmutada"
  ● embedding_text: 318 chars → ~1800 chars (fuente: PDF págs. 6-7)
  [referencias ya son genéricas — sin cambios]

HW_004  (BIOS / CMOS — pila CR2032)
  ● modelo_equipo : "PC genérico (placas ATX/mATX)" — ya genérico, sin cambios
  ● embedding_text: 315 chars → ~1900 chars (fuente: PDF págs. 8-10)
  [todos los demás campos ya son genéricos — sin cambios]

HW_005  (Throttling — diagnóstico avanzado)
  ● modelo_equipo : "PC genérico (escritorio y portátil)" — ya genérico, sin cambios
  ● embedding_text: 306 chars → ~1950 chars (fuente: PDF págs. 11-13)
  [todos los demás campos ya son genéricos — sin cambios]

HW_006  (USB multiboot de diagnóstico)
  ● modelo_equipo : "PC genérico" — ya genérico, sin cambios
  ● embedding_text: 376 chars → ~1900 chars (fuente: PDF págs. 14-16)
  [todos los demás campos ya son genéricos — sin cambios]
"""

# ---------------------------------------------------------------------------
# PASO 2+3 — Diccionario de cambios
# Clave = ID de entrada. Valor = dict con solo los campos que cambian.
# Listas de keywords/referencias se reemplazan completamente (no se hace merge).
# ---------------------------------------------------------------------------

CAMBIOS: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════
    # HW_001 — Sobrecalentamiento: pasta térmica degradada y polvo
    # Fuente PDF: páginas 1-3
    # ══════════════════════════════════════════════════════════════════════
    "HW_001": {
        "modelo_equipo": "Laptops en general",

        "keywords": [
            "sobrecalentamiento",
            "polvo",
            "pasta térmica",
            "Arctic MX-4",
            "Arctic MX-6",
            "Thermal Grizzly Kryonaut",
            "Noctua NT-H1",
            "soplador eléctrico",
            "brocha antiestática",
            "throttling",
            "Pump-out Effect",
            "TjMax",
            "HWiNFO64",
            "Cinebench R23",
            "Delta T",
        ],

        "referencias": [
            "Hoja de datos Arctic MX-4 / MX-6",
            "Hoja de datos Thermal Grizzly Kryonaut",
            "Documentación HWiNFO64 — sensores y banderas de throttling",
        ],

        "embedding_text": (
            "El sobrecalentamiento en laptops es uno de los fallos más frecuentes después de cierto tiempo de uso "
            "y se origina principalmente por la acumulación progresiva de polvo, pelusa y partículas dentro del "
            "ventilador, las rejillas de ventilación y el disipador de calor, que actúa como barrera aislante y "
            "reduce la disipación térmica. Simultáneamente, la pasta térmica entre el procesador y el disipador "
            "se degrada con el tiempo: se seca, se cuartea y pierde conductividad, proceso que se acelera por el "
            "Pump-out Effect, fenómeno en que los ciclos repetidos de expansión y contracción térmica desplazan "
            "gradualmente la pasta fuera del área del die. El uso del equipo sobre superficies blandas que "
            "obstruyen la entrada de aire inferior, ambientes con alto polvo o ausencia total de mantenimiento "
            "durante periodos prolongados agravan el problema. Los síntomas más claros son temperaturas de CPU "
            "superiores a 90-95 °C en cargas moderadas, ventiladores operando a máximas RPM de forma constante, "
            "thermal throttling con caídas bruscas de frecuencia detectables en HWiNFO64 mediante las banderas "
            "de throttling por núcleo, y en casos extremos, apagados de emergencia (thermal shutdown) al superar "
            "el TjMax del procesador (100 °C en Intel, 95 °C en muchos AMD Ryzen móviles). El diagnóstico se "
            "realiza con herramientas como HWMonitor, Core Temp, HWiNFO64 o AIDA64, registrando temperaturas "
            "en reposo (normal: 40-65 °C) y bajo carga sostenida con Cinebench R23 o Prime95. En HWiNFO64 se "
            "analizan el diferencial térmico (Delta T), los límites de potencia PL1 y PL2, el Package Power y "
            "las banderas de throttling por núcleo. La solución requiere desarmar la laptop, bloquear "
            "mecánicamente las aspas del ventilador para evitar corriente inversa, limpiar polvo con soplador "
            "eléctrico y brochas antiestáticas, eliminar la pasta vieja con alcohol isopropílico al 99 % y "
            "aplicar pasta de calidad como Arctic MX-4, Arctic MX-6, Thermal Grizzly Kryonaut o Noctua NT-H1, "
            "usando el método de gota central o línea horizontal. El mantenimiento preventivo se recomienda cada "
            "18-24 meses en ambientes normales, cada 10-12 meses con mascotas o polvo elevado, y cada 8-12 "
            "meses en laptops gaming. No atender el problema deriva en throttling sostenido, mayor consumo "
            "energético, reducción de la vida útil de CPU, GPU y VRMs, y en casos extremos, fractura de las "
            "microesferas de soldadura BGA por fatiga termomecánica crónica."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # HW_002 — Mantenimiento en laptops con GPU dedicada (flex cables)
    # Fuente PDF: páginas 4-5
    # ══════════════════════════════════════════════════════════════════════
    "HW_002": {
        "modelo_equipo": "Laptops con GPU dedicada",

        "keywords": [
            "flex cable",
            "disipador chipset",
            "pasta térmica",
            "almohadilla térmica",
            "mantenimiento preventivo",
            "GPU dedicada",
            "calor excesivo",
            "Pump-out Effect",
            "desarme laptop",
            "HWiNFO64",
            "Arctic MX-4",
            "Arctic MX-6",
            "Thermal Grizzly Kryonaut",
            "spudger",
        ],

        # Solo el índice [1] (base-0) cambia; el resto permanece idéntico.
        # Aplicamos el reemplazo textual sobre la lista completa conservando los demás pasos.
        "pasos_solucion": [
            "Apagar y desconectar la alimentación.",
            "Retirar los tornillos de la tapa inferior siguiendo la secuencia indicada en el chasis o en el manual de servicio del modelo.",
            "Desconectar batería antes de proseguir.",
            "Liberar flex cables de pantalla y WiFi con palanca de plástico; no tirar.",
            "Retirar tornillos del disipador de CPU y chipset en orden cruzado.",
            "Limpiar polvo del disipador y ventilador con soplador.",
            "Aplicar pasta térmica fresca en CPU y chipset discretamente.",
            "Reinstalar disipador; verificar que los flex cables queden sin tensión.",
            "Cerrar tapa y confirmar que los flex no quedan pinzados.",
            "Encender y verificar temperaturas con GPU-Z y HWMonitor.",
        ],

        "referencias": [
            "Hoja de datos Arctic MX-4 / MX-6",
            "Hoja de datos Thermal Grizzly Kryonaut",
            "Guía general de mantenimiento preventivo de laptops con GPU dedicada",
        ],

        "embedding_text": (
            "El mantenimiento preventivo de laptops con GPU dedicada presenta desafíos adicionales frente a "
            "equipos más sencillos, debido a que el sistema de refrigeración es compartido entre CPU, GPU y "
            "chipset, y el acceso interno requiere un desarme más cuidadoso para no dañar componentes delicados. "
            "Las fallas más comunes en este tipo de equipos incluyen temperaturas elevadas persistentes en CPU y "
            "GPU, throttling térmico que afecta tanto al procesador como a la tarjeta gráfica, ventiladores "
            "ruidosos incluso en reposo, calor localizado en zonas específicas del chasis, e inestabilidad del "
            "sistema durante cargas gráficas prolongadas. Estas fallas se originan por la obstrucción del flujo "
            "de aire con polvo compactado en ventiladores, heatpipes y disipadores, la degradación de las "
            "interfaces térmicas en CPU, GPU y chipset, y el Pump-out Effect que desplaza gradualmente la pasta "
            "térmica fuera del área de contacto. El diagnóstico combina monitorización de temperaturas con "
            "HWiNFO64, HWMonitor o AIDA64, pruebas de estrés para identificar si el throttling es temprano o "
            "existen diferencias térmicas significativas entre componentes, e inspección visual de rejillas y "
            "auscultación del ventilador para detectar obstrucción severa o desgaste mecánico. La solución "
            "efectiva requiere desarmar la laptop de forma controlada: identificar y retirar tornillos "
            "correctamente (muchos ocultos bajo almohadillas), desconectar la batería interna por seguridad, y "
            "liberar con extrema precaución los flex cables de pantalla, teclado y touchpad, que son delicados "
            "y pueden dañarse irremediablemente durante el desarme. Una vez abierto el chasis, se limpia el "
            "ventilador y los disipadores con sopladores eléctricos y brochas antiestáticas, y se reemplazan "
            "todas las interfaces térmicas: pasta térmica en CPU y GPU con opciones como Arctic MX-4, MX-5, "
            "MX-6 o Thermal Grizzly Kryonaut, y almohadillas térmicas en chipset, VRMs y memorias, respetando "
            "el grosor original (normalmente entre 0,5 mm y 2,0 mm). El mantenimiento preventivo se recomienda "
            "cada 12-18 meses en condiciones normales y cada 8-12 meses en equipos gaming o con uso intensivo. "
            "Un desarme incorrecto puede generar rotura de flex cables, pines doblados o pérdida de garantía; "
            "no atender el mantenimiento conduce a reducción del rendimiento, ruido excesivo y acortamiento de "
            "la vida útil de los componentes por estrés térmico sostenido."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # HW_003 — Fuente de alimentación en monitores (capacitores)
    # Fuente PDF: páginas 6-7
    # ══════════════════════════════════════════════════════════════════════
    "HW_003": {
        "modelo_equipo": "Monitores LCD/LED",

        "keywords": [
            "monitor",
            "capacitor hinchado",
            "ESR",
            "fuente interna",
            "fuente conmutada",
            "MOSFET",
            "driver backlight",
            "parpadeo",
            "no enciende",
            "diagnóstico energía",
            "soldadura fría",
            "IPC-7711",
        ],

        "embedding_text": (
            "Los problemas de alimentación en monitores LCD y LED se originan principalmente por la degradación "
            "de los capacitores electrolíticos ubicados en la fuente de poder interna conmutada, especialmente "
            "en la etapa primaria y secundaria. Este tipo de falla es muy común en monitores después de varios "
            "años de uso continuo y suele manifestarse de forma progresiva. Los síntomas más frecuentes incluyen "
            "que el monitor no enciende en absoluto, enciende pero se apaga después de unos segundos, presenta "
            "parpadeo intermitente de la imagen, emite un clic repetitivo al intentar encender, genera olor a "
            "componentes quemados o sobrecalentados, o muestra una imagen inestable que aparece y desaparece. "
            "Con el tiempo, los capacitores pierden capacidad, aumentan su resistencia interna (ESR, Equivalent "
            "Series Resistance) y pueden hincharse visiblemente en la parte superior. Otros componentes que "
            "suelen fallar son los fusibles de entrada, los diodos rectificadores, los transistores de potencia "
            "(MOSFETs) y los transformadores o bobinas. El deterioro se acelera por el uso prolongado sin "
            "periodos de reposo, fluctuaciones frecuentes en el voltaje de red, ambientes con alta temperatura "
            "o humedad, y el uso de protectores de sobretensión de baja calidad. El diagnóstico requiere "
            "inspección visual de la placa buscando capacitores con la parte superior abombada, fugas de "
            "electrolito o goma de sellado deteriorada; mediciones con multímetro en modo continuidad para el "
            "fusible y en modo voltaje para verificar las etapas; y medición de ESR con un medidor "
            "especializado, ya que un capacitor puede tener capacitancia cercana al nominal pero un ESR elevado "
            "que lo hace defectuoso. También es útil realizar pruebas de encendido con la fuente parcialmente "
            "desconectada del panel para aislar si el problema está en la fuente o en el controlador del "
            "backlight (inversor o driver LED). La reparación más común es el reemplazo de capacitores "
            "defectuosos respetando capacitancia, voltaje de trabajo y polaridad, y la revisión de soldaduras "
            "frías en componentes de potencia. Es fundamental tomar precauciones de seguridad: los capacitores "
            "de la etapa primaria pueden mantener carga peligrosa varios minutos después de desconectar el "
            "equipo. Como prevención se recomienda usar protectores de sobretensión o UPS de calidad, evitar "
            "humedad o temperatura elevadas, y realizar revisiones periódicas en monitores de más de cinco años "
            "de uso intensivo. Las reparaciones de nivel electrónico deben seguir la norma IPC-7711/7721 para "
            "garantizar la calidad de las soldaduras y la integridad de la placa."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # HW_004 — BIOS corrupta / pila CR2032 agotada
    # Fuente PDF: páginas 8-10
    # ══════════════════════════════════════════════════════════════════════
    "HW_004": {
        # modelo_equipo ya es genérico — no se toca.
        # keywords, herramientas, pasos, referencias ya son genéricos — no se tocan.

        "embedding_text": (
            "Los problemas de arranque relacionados con la corrupción de la configuración BIOS/UEFI o el "
            "agotamiento de la pila CMOS son relativamente comunes en equipos que llevan varios años sin "
            "mantenimiento o que han sufrido cortes de energía frecuentes. La memoria CMOS, mantenida activa "
            "por una pila CR2032 de respaldo, almacena la configuración de la placa base (secuencia de arranque, "
            "parámetros de memoria, fecha y hora) cuando el equipo está desconectado de la corriente. Cuando "
            "esta pila pierde su capacidad de retención (normalmente entre 4 y 7 años), la configuración se "
            "pierde o se corrompe. Los síntomas más característicos incluyen que el equipo no completa el POST "
            "(Power-On Self-Test), emite beep codes repetitivos según el fabricante de la BIOS (AMI, Award, "
            "Phoenix), la fecha y hora se resetean a valores por defecto (año 2000 o 2010) en cada arranque, "
            "el equipo no reconoce los dispositivos de almacenamiento o la secuencia de arranque configurada, "
            "y la BIOS no guarda los cambios aunque se guarden y se reinicie el sistema. El diagnóstico comienza "
            "por identificar el patrón de beep codes y consultar la documentación del fabricante, y por medir "
            "el voltaje de la pila CR2032 con multímetro: una pila en buen estado debe entregar 3,0 V o más; "
            "por debajo de 2,8 V es necesario reemplazarla. También es indicativo que los cambios de "
            "configuración de BIOS no se retengan tras el reinicio. La solución más directa es realizar un "
            "reset completo de la configuración BIOS mediante el jumper CLR_CMOS (o CMOS_CLR), puenteando los "
            "pines indicados durante 10-15 segundos con el equipo apagado y desconectado. Alternativamente, se "
            "puede retirar la pila durante 5-10 minutos para descargar completamente los capacitores de la "
            "placa. Si el problema persiste, se reemplaza la pila CR2032 por una nueva de calidad, respetando "
            "la polaridad. En casos donde el reset y el cambio de pila no resuelven la falla, puede ser "
            "necesario actualizar o volver a flashear la BIOS usando la utilidad del fabricante (EZ Flash, "
            "M-Flash, Q-Flash) desde un USB formateado en FAT32 con el archivo de BIOS sin renombrar. Este "
            "proceso debe realizarse con una fuente de poder estable y sin interrupciones, ya que una "
            "actualización fallida puede dejar la placa inoperativa (brick). Como prevención se recomienda "
            "cambiar la pila CMOS cada 5-6 años, documentar la configuración de BIOS antes de actualizaciones, "
            "y usar un UPS en entornos con cortes de energía frecuentes para reducir el riesgo de corrupción "
            "de la configuración UEFI durante el proceso de flasheo."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # HW_005 — Throttling térmico, por potencia y por corriente
    # Fuente PDF: páginas 11-13
    # ══════════════════════════════════════════════════════════════════════
    "HW_005": {
        # modelo_equipo ya es genérico — no se toca.
        # keywords, herramientas, pasos, referencias ya son genéricos — no se tocan.

        "embedding_text": (
            "El throttling es un mecanismo de autoprotección que activa el procesador o la GPU al detectar "
            "condiciones que superan sus límites de diseño, reduciendo automáticamente la frecuencia de "
            "operación y el voltaje para disminuir la generación de calor o el consumo de energía. Existen "
            "tres tipos principales: throttling térmico, que se activa cuando la temperatura supera el TjMax "
            "(generalmente 100 °C en Intel, 95 °C en AMD Ryzen móviles); throttling por potencia (Power Limit "
            "Throttling), que ocurre cuando el procesador alcanza los límites PL1 y PL2 en Intel, o PPT en AMD, "
            "muy común en laptops donde el adaptador tiene capacidad limitada; y throttling por corriente, menos "
            "frecuente, que se produce cuando la demanda de corriente supera la capacidad de la fuente o de las "
            "fases de poder (VRMs) de la placa base. Los síntomas característicos son caídas bruscas de "
            "rendimiento durante tareas exigentes, temperaturas de CPU o GPU superiores a 90-95 °C de forma "
            "sostenida, ventiladores a máxima velocidad durante periodos prolongados, y la percepción de que el "
            "equipo pierde rendimiento progresivamente conforme sube la temperatura. El diagnóstico preciso "
            "requiere HWiNFO64, que permite observar en tiempo real temperaturas, frecuencias por núcleo, "
            "consumo de energía (Package Power), límites PL1 y PL2, y las banderas de throttling activas. "
            "Complementariamente se usan HWMonitor, Core Temp y ThrottleStop para Intel. Para reproducir el "
            "problema se ejecutan pruebas de estrés como Cinebench R23 Multi Core, Prime95 o AIDA64 durante al "
            "menos 10 minutos. Si la frecuencia cae significativamente mientras la temperatura supera el TjMax, "
            "es throttling térmico; si la frecuencia baja pero la temperatura es moderada, es throttling por "
            "límite de potencia. La solución varía según el tipo: para throttling térmico, limpieza profunda "
            "del sistema de refrigeración y reemplazo de pasta térmica; para throttling por potencia en laptops, "
            "ajuste controlado de PL1 y PL2 con ThrottleStop o AMD Ryzen Master dentro de los límites del "
            "fabricante; en algunos casos undervolting controlado para reducir voltaje y calor sin sacrificar "
            "rendimiento. En escritorio se puede mejorar el cooler de CPU, agregar ventiladores al gabinete o "
            "usar refrigeración líquida; se recomienda presión de aire positiva (más entrada que salida) para "
            "reducir la temperatura ambiental interna. Como prevención se recomienda mantenimiento periódico "
            "del sistema de refrigeración cada 12-18 meses, evitar superficies que bloqueen la ventilación, "
            "mantener actualizados los controladores de chipset y la BIOS/UEFI, y monitorear periódicamente "
            "las temperaturas. Sin corrección, el throttling sostenido reduce el rendimiento, acelera el "
            "envejecimiento de los componentes y acorta la vida útil a largo plazo."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # HW_006 — USB multiboot de diagnóstico (Ventoy, MemTest86, Victoria)
    # Fuente PDF: páginas 14-16
    # ══════════════════════════════════════════════════════════════════════
    "HW_006": {
        # modelo_equipo ya es genérico — no se toca.
        # keywords, herramientas, pasos, referencias ya son genéricos — no se tocan.

        "embedding_text": (
            "Un USB multiboot de diagnóstico permite arrancar cualquier computadora en un entorno limpio e "
            "independiente del sistema operativo instalado para ejecutar herramientas especializadas que "
            "identifican fallas de hardware difíciles de reproducir o diagnosticar desde el SO afectado. Esta "
            "aproximación es fundamental cuando el equipo presenta pantallazos azules (BSOD) aleatorios, "
            "congelamientos intermitentes, lentitud extrema sin causa aparente, errores de memoria reportados "
            "por Windows, o cuando el sistema operativo no arranca. La herramienta Ventoy simplifica "
            "enormemente la creación de USB multiboot: convierte un USB convencional en un dispositivo "
            "multiboot sin necesidad de formatearlo cada vez que se añade una nueva herramienta; crea una "
            "partición visible donde se almacenan múltiples archivos ISO, y al arrancar presenta un menú "
            "gráfico para seleccionar la herramienta deseada. El proceso de creación comienza descargando "
            "Ventoy desde su sitio oficial, ejecutando Ventoy2Disk.exe, verificando dos veces que se "
            "selecciona el dispositivo correcto, y procediendo con la instalación. Las ISOs se copian "
            "directamente a la partición creada sin necesidad de descomprimirlas. Entre las herramientas más "
            "recomendadas están: MemTest86, la más confiable para pruebas exhaustivas de memoria RAM a bajo "
            "nivel, detectando errores que el SO no identifica; se recomiendan al menos dos pasadas completas "
            "(aproximadamente 40 min por pasada). Victoria HDD y MHDD para pruebas de superficie en discos "
            "duros mecánicos y SSDs, capaces de detectar sectores defectuosos (bad blocks), sectores inestables "
            "y problemas de lectura/escritura; Victoria muestra mapas visuales del estado del disco. "
            "CrystalDiskInfo para revisar el estado de salud del disco mediante los atributos S.M.A.R.T. antes "
            "de pruebas más invasivas, identificando sectores reasignados y temperatura elevada. HWiNFO64 para "
            "monitoreo avanzado de temperaturas, voltajes y consumo durante las pruebas. El orden recomendado "
            "es: primero revisar S.M.A.R.T. con CrystalDiskInfo, luego pruebas de memoria con MemTest86, y "
            "después pruebas de superficie con Victoria HDD. En MemTest86 cualquier error indica RAM defectuosa; "
            "en Victoria, bloques rojos son sectores irrecuperables y bloques amarillos son inestables. El "
            "acúmulo progresivo de sectores reasignados en S.M.A.R.T. señala que el disco debe reemplazarse "
            "antes de que falle completamente. Es fundamental hacer backup de datos importantes antes de "
            "pruebas de escritura en disco. Mantener el USB actualizado con versiones recientes garantiza "
            "compatibilidad con hardware moderno y mayor precisión en la detección de errores."
        ),
    },
}


# ---------------------------------------------------------------------------
# Lógica de aplicación
# ---------------------------------------------------------------------------

def _backup(kb_path: Path) -> Path:
    """Crea una copia de seguridad con timestamp y retorna la ruta del backup."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup  = BACKUP_DIR / f"hardware_knowledge_base_{ts}.json"
    shutil.copy2(kb_path, backup)
    print(f"  [BACKUP] → {backup}")
    return backup


def _apply_cambios(entry: dict, cambios_entry: dict) -> list[str]:
    """
    Aplica cambios a una entrada y retorna lista de strings describiendo cada cambio.
    Modifica `entry` in-place.
    """
    log: list[str] = []
    for campo, nuevo_valor in cambios_entry.items():
        old = entry.get(campo)

        if campo == "pasos_solucion":
            # Reemplaza la lista completa; muestra solo el paso que cambió.
            old_paso = (old or [])[1] if old and len(old) > 1 else "(vacío)"
            entry[campo] = nuevo_valor
            new_paso = nuevo_valor[1]
            log.append(f"    pasos_solucion[1]:\n"
                       f"      ANTES : {old_paso!r}\n"
                       f"      AHORA : {new_paso!r}")

        elif campo == "keywords":
            entry[campo] = nuevo_valor
            removed = set(old or []) - set(nuevo_valor)
            added   = set(nuevo_valor) - set(old or [])
            if removed:
                log.append(f"    keywords  ELIMINADOS : {sorted(removed)}")
            if added:
                log.append(f"    keywords  AÑADIDOS   : {sorted(added)}")

        elif campo == "referencias":
            entry[campo] = nuevo_valor
            removed = set(old or []) - set(nuevo_valor)
            added   = set(nuevo_valor) - set(old or [])
            if removed:
                log.append(f"    referencias ELIMINADAS:\n"
                           + "\n".join(f"      - {r}" for r in sorted(removed)))
            if added:
                log.append(f"    referencias AÑADIDAS:\n"
                           + "\n".join(f"      + {r}" for r in sorted(added)))

        elif campo == "embedding_text":
            old_len = len(old or "")
            entry[campo] = nuevo_valor
            new_len = len(nuevo_valor)
            log.append(f"    embedding_text : {old_len} chars → {new_len} chars")

        else:
            entry[campo] = nuevo_valor
            log.append(f"    {campo:<20}: {old!r}  →  {nuevo_valor!r}")

    return log


def main() -> None:
    print(PLAN_TEXT)

    if not KB_PATH.exists():
        print(f"ERROR: No se encontró la Knowledge Base en:\n  {KB_PATH.resolve()}")
        sys.exit(1)

    # ── Backup ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("PASO 4 — Aplicando cambios con backup previo")
    print("=" * 72)
    _backup(KB_PATH)

    # ── Carga ────────────────────────────────────────────────────────────
    with open(KB_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    entries    = data["hardware_entries"]
    n_original = len(entries)
    print(f"\n  KB cargada: {n_original} entradas\n")

    # ── Aplicar cambios ─────────────────────────────────────────────────
    aplicados = 0
    for entry in entries:
        entry_id = entry["id"]
        if entry_id not in CAMBIOS:
            print(f"  {entry_id} — sin cambios (campos ya genéricos)")
            continue

        print(f"  {entry_id} — aplicando {len(CAMBIOS[entry_id])} campo(s):")
        log_lines = _apply_cambios(entry, CAMBIOS[entry_id])
        for line in log_lines:
            print(line)
        aplicados += 1
        print()

    # ── Guardar ─────────────────────────────────────────────────────────
    with open(KB_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"  Guardado: {KB_PATH.resolve()}")
    print(f"  Entradas modificadas: {aplicados} / {n_original}")
    print("=" * 72)


if __name__ == "__main__":
    main()
