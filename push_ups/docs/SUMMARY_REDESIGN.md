# Documentación de Cambios: Rediseño de Resúmenes y HUD en Vivo (Push-ups)

Este documento detalla la reorganización y mejoras visuales implementadas en el sistema de análisis y renderizado de video de flexiones (`push_ups`), abarcando tanto el formato **Horizontal (1920×1080)** como el formato **Reel Vertical (1080×1920)**.

---

## 1. Resumen de Objetivos y Problemas Resueltos

| Componente | Problema Previo | Solución Implementada |
| :--- | :--- | :--- |
| **Resumen Horizontal (1080p)** | Para series > 25 reps, la tabla de 3 columnas colisionaba físicamente con el gráfico de barras (`x = 1100..1426`). | Arquitectura de **dos paneles limpios** (Izquierdo: KPIs, fatiga, bloques; Derecho: Gráfico de barras de 580px y métricas de tempo). Cero solapamiento. |
| **Resumen Reel (Vertical)** | Desbordamiento fuera de pantalla y saturación en zonas de interacción de redes sociales. | Tarjeta central unificada (`x: 60..1020`, `y: 180..1740`) centrada en el eje horizontal (`x = 540`) con KPIs en 2×2, gráfico de barras ancho y diagnóstico. |
| **Fondo Fantasma (Ghosting)** | El fondo oscuro del fotograma congelado final retenía números y curvas del último fotograma del HUD en vivo. | Captura limpia de `closing_base` antes de superponer el HUD interactivo durante la reproducción. |
| **Barras Acumulativas en Vivo** | Fila de mini-barras (`_bars`) llenándose rep por rep durante el video, saturando la pantalla y redundante con el resumen final. | **Eliminadas** en vivo en ambos formatos. En Horizontal se sustituyó por una tabla clara de "Últimas repeticiones" (6 filas). En Reel se maximizó el espacio de la curva senoidal y métricas de tempo. |
| **Disposición del HUD en Reel** | Cajas flotantes desordenadas que invadían el torso y las piernas del atleta durante la ejecución. | **HUD dividido en 2 extremos centrados**: tarjeta superior pegada arriba (`y: 50..190`) con tiempo centrado, tarjeta inferior pegada abajo (`y: 1450..1860`), dejando 1260px verticales en el centro 100% libres para el atleta. |
| **Textos de Auditoría al Pie** | Mensajes redundantes (*"detalle completo de rep por rep disponible en reps.json y en interfaz web (127.0.0.1:8080)"*) en el video final. | **Eliminados** de las pantallas de cierre de ambos formatos, conservando íntegras todas las tarjetas de métricas (`SUBIDA PROM.`, `BAJADA PROM.`, `PAUSA PROM.`, etc.). |

---

## 2. Telemetría en Vivo durante la Reproducción

### A. Formato Reel Vertical (1080×1920)

El HUD en vivo se estructura en dos tarjetas fijas centradas horizontalmente (`x = 540`, ancho 960px):

```
+-------------------------------------------------------------------+
|  y: 0..50   Margen superior                                       |
|  +-------------------------------------------------------------+  |
|  | [y: 50..190] TARJETA SUPERIOR CENTRADA                      |  |
|  | FLEXIONES 19/50            TIEMPO 00:18     [REP DETECTADA] |  |
|  +-------------------------------------------------------------+  |
|                                                                   |
|                                                                   |
|                  ZONA LIBRE DEL ATLETA                            |
|                  y: 190 -> 1450 (1260 px)                         |
|             (Visión 100% despejada del cuerpo)                    |
|                                                                   |
|                                                                   |
|  +-------------------------------------------------------------+  |
|  | [y: 1450..1860] TARJETA INFERIOR CENTRADA                   |  |
|  | Rep 19  ·  Profundidad -0,14  ·  Grado [ROM ALTO]           |  |
|  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |  |
|  | Curva senoidal de profundidad en tiempo real                |  |
|  |                                                             |  |
|  | [SUBIDA 0,31 s]        [BAJADA 0,52 s]        [PAUSA 0,28 s]|  |
|  +-------------------------------------------------------------+  |
|  y: 1860..1920 Margen inferior                                    |
+-------------------------------------------------------------------+
```

- **Top HUD Card** (`top_box = (60, 50, 1020, 190)`):
  - Izquierda: Contador actual/objetivo (`FLEXIONES 19/50`).
  - Centro: Temporizador acumulado (`TIEMPO 00:18`).
  - Derecha: Badge dinámico de repetición (`[REP DETECTADA]`).
- **Bottom HUD Card** (`bot_box = (60, 1450, 1020, 1860)`):
  - Línea de estado de repetición actual.
  - Gráfico de onda senoidal de profundidad en tiempo real con líneas de umbral.
  - 3 tarjetas de tempo: Subida, Bajada y Pausa.
  - *Sin barras acumulativas `_bars`*.

### B. Formato Horizontal (1920×1080)

- HUD interactivo lateral que mantiene el stream de video a la izquierda/centro.
- Gráfico senoidal en tiempo real y métricas de tempo en el panel derecho.
- Tabla de "Últimas repeticiones" (máx. 6 filas) con actualización progresiva en vivo, sustituyendo el gráfico de mini-barras acumulativas.

---

## 3. Pantallas de Cierre (Resumen Final de la Serie)

### A. Formato Horizontal (1920×1080)

Arquitectura de **Dos Paneles Independientes** sobre fondo oscuro limpio al 94%:

```
+----------------------------------------------------------------------------------------------------+
|  PANEL IZQUIERDO (x: 80 -> 840, w: 760)              PANEL DERECHO (x: 900 -> 1840, w: 940)        |
|                                                                                                    |
|  Resumen de la serie                                 Profundidad por rep                            |
|  50 flexiones · 55 s                                 Consistencia del movimiento en la serie        |
|  [35 ROM ALTO] [12 ROM FULL] [3 BAJO MÍN.]           +-------------------------------------------+  |
|                                                      |  -- Alto (-0,18)                          |  |
|  +------------+ +------------+                       |  -- Semi (-0,12)    ||||||                |  |
|  | DURACIÓN   | | RITMO      |                       |  -- Mín. (-0,06)    ||||||||||||||        |  |
|  | 55 s       | | 1 rep/1,1s |                       |                     ||||||||||||||||||||| |  |
|  +------------+ +------------+                       +-------------------------------------------+  |
|  | PROF. PROM.| | MEJOR REP  |                         1   5  10  15  20  25  30  35  40  45  50    |
|  | -0,13      | | #48 -0,04  |                                                                      |
|  +------------+ +------------+                       +-------------+ +-------------+ +-------------+ |
|                                                      |SUBIDA PROM. | |BAJADA PROM. | |PAUSA PROM.  | |
|  DIAGNÓSTICO DE FATIGA                               |0,31 s       | |0,50 s       | |0,30 s       | |
|  • Profundidad: -0,19 inicial -> -0,08 final         +-------------+ +-------------+ +-------------+ |
|  • Subida: 0,34 s -> 0,49 s (+44% más lento)                                                        |
|  • Pausa más larga: 1,0 s antes de la rep 50                                                        |
|                                                                                                     |
|  EVOLUCIÓN POR BLOQUES (DECENAS)                                                                    |
|  Reps 01–10: Prof. -0,18 · Subida 0,32 s                                                            |
|  Reps 11–20: Prof. -0,16 · Subida 0,32 s                                                            |
|  Reps 21–30: Prof. -0,12 · Subida 0,31 s                                                            |
|  Reps 31–40: Prof. -0,12 · Subida 0,26 s                                                            |
|  Reps 41–50: Prof. -0,07 · Subida 0,36 s                                                            |
+----------------------------------------------------------------------------------------------------+
```

### B. Formato Reel Vertical (1080×1920)

Tarjeta central centrada (`box = (60, 180, 1020, 1740)`, ancho 960px, altura 1560px):

```
+-------------------------------------------------------------+
|                                                             |
|   +-----------------------------------------------------+   |
|   |  FLEXIONES · RESUMEN FINAL                          |   |
|   |  50 REPS · 00:55                                    |   |
|   |  [35 ROM ALTO]  [12 ROM FULL]  [3 BAJO MÍNIMO]      |   |
|   |                                                     |   |
|   |  +-----------------------+ +---------------------+  |   |
|   |  | PROF. PROMEDIO        | | MEJOR REP           |  |   |
|   |  | -0,13                 | | #48 (-0,04)         |  |   |
|   |  +-----------------------+ +---------------------+  |   |
|   |  | RITMO                 | | SUBIDA PROM.        |  |   |
|   |  | 1 rep / 1,1 s         | | 0,31 s              |  |   |
|   |  +-----------------------+ +---------------------+  |   |
|   |                                                     |   |
|   |  PROFUNDIDAD POR REP                                |   |
|   |  +-----------------------------------------------+  |   |
|   |  | -- Alto                                       |  |   |
|   |  | -- Semi     ||||||||||||||||||||||            |  |   |
|   |  | -- Mín.     ||||||||||||||||||||||||||||||||| |  |   |
|   |  +-----------------------------------------------+  |   |
|   |    1    10    20    30    40    50                  |   |
|   |                                                     |   |
|   |  DIAGNÓSTICO DE FATIGA                              |   |
|   |  • Profundidad: -0,19 inicial -> -0,08 final        |   |
|   |  • Subida: 0,34 s -> 0,49 s (+44% más lento)        |   |
|   |  • Pausa máx.: 1,0 s antes de la rep 50             |   |
|   |                                                     |   |
|   |  EVOLUCIÓN POR BLOQUES (DECENAS)                    |   |
|   |  Reps 01–10: Prof. -0,18 · Subida 0,32 s            |   |
|   |  Reps 11–20: Prof. -0,16 · Subida 0,32 s            |   |
|   |  Reps 21–30: Prof. -0,12 · Subida 0,31 s            |   |
|   |  Reps 31–40: Prof. -0,12 · Subida 0,26 s            |   |
|   |                                                     |   |
|   |  Bajada prom: 0,50 s  ·  Pausa prom: 0,30 s         |   |
|   +-----------------------------------------------------+   |
|                                                             |
+-------------------------------------------------------------+
```

---

## 4. Archivos Modificados en el Código Fuente

### [`push_ups/pushup_demo/render.py`](file:///c:/Users/Dimitry/Desktop/vision-demos/push_ups/pushup_demo/render.py)

1. **`_summary_horizontal(...)`**:
   - Layout de dos columnas independientes (`left_panel` y `right_panel`).
   - Cálculo automático de bloques de 10 reps con `_blocks_summary(...)`.
   - Gráfico de barras de profundidad con altura de 580px y guía de calibración.
   - Eliminación de la leyenda inferior sobre `reps.json` y el servidor local, manteniendo las 3 tarjetas de métricas.

2. **`_summary_reel(...)`**:
   - Tarjeta centrada en coordenadas `(60, 180, 1020, 1740)`.
   - Grid 2×2 de KPIs clave.
   - Gráfico de barras con ancho completo (880px) y altura de 240px.
   - Eliminación del texto de pie de página de auditoría.

3. **`_reel_frame(...)`**:
   - Reubicación de tarjeta superior a `(60, 50, 1020, 190)` con `TIEMPO` centrado.
   - Reubicación de tarjeta inferior a `(60, 1450, 1020, 1860)` con curva senoidal y métricas de tempo.
   - Eliminación del llamado a `_bars` en vivo.

4. **`_horizontal_frame(...)`**:
   - Eliminación del llamado a `_bars` en vivo.
   - Inclusión de tabla progresiva de historial reciente de 6 repeticiones.

5. **Captura de `closing_base`**:
   - Guardado del fotograma antes de aplicar las capas del HUD interactivo, evitando el efecto fantasma.

---

## 5. Plataforma Web y Auditoría Detallada

- **Servidor Web Activo**: `web_app.py` ejecutándose en `http://127.0.0.1:8080/`.
- La interfaz web aloja la tabla interactiva rep por rep, reproducción sincronizada de videos, métricas por repetición y exportación de `reps.json` y `report.txt`.
