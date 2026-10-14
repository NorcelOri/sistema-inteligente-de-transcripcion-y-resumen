# PLAN.md — Sistema inteligente de transcripción y resumen automático de audios (NLP)

Proyecto de unidad, curso Procesamiento de Lenguaje Natural, UNA Puno.
Autor: Cuyo Zamata Yimmy Yeyson. Plazo total: 6 días.

## 1. Objetivo

**Alcance:** el sistema es general: reuniones, clases, videos informativos, de cualquier duración. Los 3 videos de reuniones de la sección 3 son solo material de prueba; ninguna decisión de diseño debe depender de su contenido (nombres, siglas, temas).

Dado un archivo de audio o video (.mp3, .wav, .m4a, .mp4), el sistema:
1. Extrae y prepara el audio.
2. Lo transcribe (Whisper).
3. Limpia el texto (muletillas, normalización, oraciones).
4. Extrae palabras clave, temas y entidades (personas, fechas, lugares).
5. Genera un resumen de 150 a 300 palabras (mT5 con map-reduce; TextRank como respaldo).
6. Entrega todo en una interfaz Gradio, más un `.txt` (transcripción) y un `.pdf` (resumen).

Todo con **modelos preentrenados**. No se entrena nada.

## 2. Entorno (YA CONFIGURADO, no reinstalar)

- Windows, PowerShell. GPU: NVIDIA RTX 5070 Laptop (CUDA 12.8).
- Python 3.12.10. Entorno virtual FUERA de OneDrive: `C:\venvs\nlp-audio`
- Activar: `C:\venvs\nlp-audio\Scripts\Activate.ps1` (el prompt debe mostrar `(nlp-audio)`).
- Carpeta del proyecto: `C:\Users\User\OneDrive\Desktop\proyecto-nlp-audio`
- PyTorch 2.11.0+cu128 (CUDA verificado). **No reinstalar torch** ni dejar que otra librería lo reemplace por una versión sin CUDA.
- Instalados: openai-whisper, transformers 5.19.0, sentencepiece, spacy 3.8.16 + `es_core_news_md`, keybert, sentence-transformers, sumy, nltk (punkt, punkt_tab), gradio, fpdf2, pydub, noisereduce, jiwer, rouge-score, ipykernel, jupyter.
- ffmpeg instalado vía winget (disponible en PATH).
- Los modelos se descargan solos en `C:\Users\User\.cache` la primera vez (mT5 pesa ~2 GB).

## 3. Datos de prueba (solo para probar; el sistema no depende de ellos)

3 videos de reuniones de Google Meet (español, poco ruido, varios hablantes), con consentimiento de los participantes:

| Archivo | Duración |
|---|---|
| reunion_1 | 30 min |
| reunion_2 | 23 min |
| reunion_3 | 18 min |

- Guardarlos en `data/videos/` (esta carpeta va en `.gitignore`; no subirlos a ningún repositorio público por privacidad).
- Para la evaluación del WER: transcribir **a mano** un fragmento de 2 a 3 minutos de uno de los videos y guardarlo en `data/referencia/`.
- Para el ROUGE sobre reuniones: escribir a mano un resumen de referencia de uno de los videos.

## 4. Estructura de carpetas

```
proyecto-nlp-audio/
├── PLAN.md
├── requirements.txt
├── .gitignore
├── app.py                  # Interfaz Gradio (entrega final)
├── pipeline.py             # Función única que orquesta todas las etapas
├── src/
│   ├── __init__.py
│   ├── audio.py            # ffmpeg / pydub / noisereduce
│   ├── transcripcion.py    # Whisper
│   ├── texto.py            # limpieza, muletillas, oraciones (spaCy + regex)
│   ├── nlp.py              # KeyBERT, temas, entidades, fechas
│   ├── resumen.py          # mT5 map-reduce + TextRank (sumy)
│   ├── salidas.py          # .txt, .pdf, metadatos
│   └── evaluacion.py       # WER, ROUGE, tiempos, % reducción
├── notebooks/
│   └── pruebas.ipynb       # pruebas por etapa y capturas para el informe
├── data/
│   ├── videos/
│   ├── audios/             # .wav extraídos
│   ├── transcripciones/    # .txt (cache: transcribir cada video UNA sola vez)
│   └── referencia/
└── resultados/             # .txt y .pdf generados
```

## 5. Diseño de los módulos

Cada módulo expone funciones simples y testeables. Todas las rutas con `pathlib`.

### src/audio.py
- `extraer_audio(ruta_entrada) -> ruta_wav`: ffmpeg a WAV mono 16 kHz. Acepta .mp3, .wav, .m4a, .mp4.
- `verificar_formato(ruta) -> bool`.
- `reducir_ruido(ruta_wav) -> ruta_wav` (noisereduce). **Opcional**, activable con un parámetro; permite comparar el WER con y sin reducción de ruido.
- `normalizar_volumen(ruta_wav)`.
- `duracion_segundos(ruta) -> float`.
- La segmentación para el reconocimiento la hace Whisper internamente (ventanas de 30 s); documentarlo en el informe.

### src/transcripcion.py
- `cargar_whisper(modelo="small", device="cuda")`: cargar una sola vez y reutilizar (cache en variable global o clase).
- `transcribir(ruta_wav, idioma="es") -> dict` con texto completo y segmentos con tiempos. Usar `fp16=True` en GPU.
- Guardar el resultado en `data/transcripciones/<nombre>.txt` y reutilizarlo si ya existe.
- Probar `small` primero; si el WER es alto, probar `medium` (la GPU alcanza).
- **Glosario opcional:** `transcribir(..., glosario=[...])`. Lista de términos (nombres propios, siglas) que da el usuario; vacía por defecto. Con ella se construye el `initial_prompt` (`construir_prompt`). Sin glosario no se usa prompt. No hay términos por defecto: el ejemplo del caso de reuniones vive en `config/ejemplos/caso_reuniones/` (`terminos.txt`, `variantes.txt`); `src/config.py` solo lee archivos si se le pasa la ruta.
- **Reparación automática** (`reparar_transcripcion`, activa por defecto): detecta (a) **bucles**: segmentos consecutivos con el mismo texto (3+ veces, o 2 veces si la frase tiene 3+ palabras) y (b) **huecos**: 8 s o más sin texto con energía de voz en el audio (50 % o más de fotogramas sobre el piso de ruido). Re-transcribe solo ese tramo (con 2 s de margen) con `condition_on_previous_text=False`; si sigue con problemas, un segundo intento sin los descartes `no_speech_threshold` y `logprob_threshold`. Semilla fija para que el resultado sea reproducible. Quita de los bordes los segmentos vecinos cuyo texto ya aparece en el tramo nuevo. Registra cada reparación (antes/después) en `resultado["reparaciones"]`.
- Los archivos `*_manual*` NO se usan para generar resultados del sistema; solo la transcripción automática con las correcciones del propio programa. Las referencias manuales solo sirven para evaluar (WER).

### src/texto.py
- `corregir_nombres(texto, glosario=None, variantes=None)`: dos capas, ambas opcionales:
  1. `corregir_glosario`: corrección aproximada con `rapidfuzz` contra el glosario del usuario, por palabra o frase (ventanas de k-1, k y k+1 palabras). Conservadora: misma letra inicial, umbral 85 (términos de 6+ letras; siglas y términos cortos solo normalizan mayúsculas), y no toca ventanas con palabras comunes del español (las 10 000 más frecuentes del vocabulario de spaCy).
  2. `corregir_variantes`: diccionario `variante => correcta` de errores ya vistos (segunda capa, opcional).
- `limpiar_texto(texto) -> str`:
  - Eliminar muletillas con regex (eh, em, este, o sea, bueno, mmm, ¿no?, etc.), sin romper palabras que las contengan (usar límites de palabra `\b`).
  - Normalizar espacios, signos y mayúsculas iniciales; quitar caracteres innecesarios.
  - Conservar la puntuación que Whisper ya genera.
- `segmentar_oraciones(texto) -> list[str]` con spaCy `es_core_news_md`.
- `tokenizar(texto) -> list[str]`.
- Guardar un ejemplo "antes / después" para el informe.

### src/nlp.py
- `extraer_palabras_clave(texto, n=10)`: KeyBERT con `paraphrase-multilingual-MiniLM-L12-v2`, `keyphrase_ngram_range=(1, 2)`, `use_mmr=True` para evitar redundancia. Usar stopwords en español (lista de spaCy).
- `extraer_temas(texto)`: dividir el texto en bloques, extraer las frases clave de cada bloque con KeyBERT y agrupar/priorizar las más frecuentes o representativas como "temas principales".
- `extraer_entidades(texto) -> dict`: spaCy NER → personas (PER), lugares (LOC), organizaciones (ORG, MISC opcional).
- `extraer_fechas(texto) -> list[str]`: expresiones regulares (formatos como "15 de octubre", "12/10/2026", "lunes", "mañana", "próxima semana", etc.).
- Deduplicar y ordenar por frecuencia. Ojo: NER sobre habla transcrita es ruidoso; filtrar entidades de un solo carácter o muy raras.

### src/resumen.py
- Modelo principal: `csebuetnlp/mT5_multilingual_XLSum`. **Cargar con `AutoTokenizer` y `AutoModelForSeq2SeqLM` directamente (no usar `pipeline`)**, porque `transformers` es versión 5.x y la API puede diferir; verificar la documentación contra la versión instalada.
- Preprocesar cada bloque con el `WHITESPACE_HANDLER` de la ficha del modelo (colapsar saltos de línea y espacios).
- Parámetros de partida: entrada `max_length=512` con truncado, `num_beams=4`, `no_repeat_ngram_size=2`, salida `max_length` ~ 84 (ajustar).
- Estrategia **map-reduce**:
  1. Dividir la transcripción en bloques de ~400 tokens, respetando límites de oración.
  2. Resumir cada bloque.
  3. Unir los resúmenes parciales y volver a resumir si exceden el tamaño objetivo.
- **Riesgo conocido:** el modelo XL-Sum fue entrenado con noticias y tiende a generar resúmenes muy cortos (estilo titular). Si no se llega a 150–300 palabras, ajustar `min_length`/`max_length`, usar bloques más pequeños (más resúmenes parciales) o combinar con TextRank. Documentar la decisión.
- Alternativa: `resumir_textrank(texto, n_oraciones)` con `sumy` (TextRank, idioma "spanish"). Debe funcionar sin GPU y ser la opción de respaldo.
- `resumir(texto, metodo="mt5" | "textrank")`.

### src/salidas.py
- `calcular_metadatos(audio, transcripcion) -> dict`: duración, número de palabras, número de oraciones.
- `guardar_txt(transcripcion, ruta)`: transcripción completa.
- `generar_pdf(resumen, palabras_clave, temas, entidades, metadatos, ruta)` con fpdf2. Cuidar la codificación del español (tildes y ñ): usar una fuente Unicode (por ejemplo DejaVu) si la fuente por defecto falla.

### src/evaluacion.py
- `wer(referencia, hipotesis)` con `jiwer` (normalizar minúsculas y puntuación antes de comparar).
- `rouge(referencia, resumen)` con `rouge-score`.
- `porcentaje_reduccion(palabras_transcripcion, palabras_resumen)`.
- Medición de tiempos por etapa (`time.perf_counter`).

### pipeline.py
- `procesar(ruta_archivo, idioma="es", reducir_ruido=False, metodo_resumen="mt5") -> dict` que encadena todas las etapas y devuelve: transcripción, resumen, palabras clave, temas, entidades, metadatos, tiempos, rutas de `.txt` y `.pdf`.
- Reporta progreso por etapa (para que la interfaz lo muestre).

### app.py (Gradio)
Interfaz con:
- Entrada: subir audio o video (.mp3, .wav, .m4a, .mp4), selector de idioma (por defecto español), casilla "reducir ruido", selector de método de resumen (mT5 / TextRank), botón "Procesar".
- Salida: pestañas o secciones para Transcripción, Resumen, Palabras clave y temas, Información relevante (personas, fechas, lugares), Metadatos y tiempos.
- Descarga: archivo `.txt` y archivo `.pdf`.
- Barra de progreso por etapa.
- Verificar la versión de Gradio instalada y usar su API (cambia entre versiones mayores).
- Lanzar en `127.0.0.1` (local).

## 6. Plan de trabajo por día

| Día | Tareas | Criterio de terminado |
|---|---|---|
| 1 | Crear estructura, `.gitignore`, `src/audio.py`, `src/transcripcion.py`. Extraer audio de un video y transcribir 2 minutos con Whisper `small`. | Transcripción de prueba guardada en `data/transcripciones/`; GPU en uso. |
| 2 | Transcribir los 3 videos completos y guardarlos en `.txt`. Implementar `src/texto.py` (muletillas, oraciones). Implementar entidades y fechas. | 3 transcripciones en disco; ejemplo antes/después de limpieza; entidades detectadas. |
| 3 | `src/nlp.py`: palabras clave y temas con KeyBERT. Revisar calidad con los 3 videos. | Palabras clave y temas coherentes con el contenido real de cada reunión. |
| 4 | `src/resumen.py`: mT5 map-reduce y TextRank. Comparar ambos en los 3 videos; ajustar parámetros. | Resumen de 150–300 palabras por video (o justificación documentada). |
| 5 | `src/salidas.py` (.txt, .pdf), `pipeline.py` y `app.py` (Gradio). | Flujo completo desde la interfaz hasta los archivos descargables. |
| 6 | Evaluación (WER, ROUGE, tiempos, % reducción), capturas, pruebas finales con los 3 videos, ajustes al informe. | Tabla de resultados y capturas listas. |

Si hay retraso: priorizar que el flujo completo funcione con TextRank antes de pulir mT5.

## 7. Evaluación (para el informe)

- **Transcripción:** WER sobre el fragmento transcrito a mano (comparar Whisper `small` vs `medium`, y con/sin reducción de ruido, si hay tiempo).
- **Resumen:** ROUGE contra el resumen escrito a mano de un video, más revisión cualitativa en los 3 videos. Opcional: ROUGE sobre una muestra pequeña de XL-Sum en español (mide el modelo en general, no el caso de reuniones).
- **Rendimiento:** tiempo por etapa y por video; % de reducción (palabras del resumen / palabras de la transcripción).

## 8. Limitaciones a declarar en el informe

- Optimizado para español (modelos, stopwords, spaCy y reglas de limpieza son de español); otros idiomas no están evaluados.
- No hay diarización: no se distingue quién habla.
- La reparación de bucles/huecos re-transcribe con Whisper, que no es determinista del todo; el texto recuperado puede tener errores, y los tiempos de los segmentos son aproximados (± varios segundos).
- La corrección con glosario solo arregla errores cercanos en escritura al término; no recupera palabras que Whisper entendió como otra cosa.
- mT5 XL-Sum fue entrenado con noticias; en reuniones puede producir resúmenes cortos o genéricos.
- El NER de spaCy sobre habla transcrita es menos preciso que sobre texto escrito.
- Los modelos tienen entrada limitada (~512 tokens), por eso se usa map-reduce.

## 9. Reglas de trabajo para Claude Code

1. Leer este archivo antes de empezar y trabajar **un día/etapa a la vez**, confirmando con el usuario antes de pasar a la siguiente.
2. Activar siempre el entorno `C:\venvs\nlp-audio` antes de ejecutar nada. No crear otro entorno ni reinstalar PyTorch.
3. Antes de instalar una librería nueva, avisar y agregarla a `requirements.txt`.
4. Código en español claro, con comentarios breves y funciones pequeñas (el proyecto se defiende oralmente; el autor debe poder explicarlo).
5. Después de cada módulo, ejecutar una prueba rápida con datos reales de `data/` y mostrar la salida.
6. No subir ni copiar los videos fuera de la carpeta del proyecto (contienen voces de personas que dieron su consentimiento solo para este uso académico).
7. Guardar las transcripciones en disco apenas se generen, para no repetir el trabajo de GPU.
8. Si una versión instalada difiere de lo que muestra este plan (transformers 5.x, gradio), consultar la documentación de la versión instalada en lugar de asumir.

## 10. Primer prompt sugerido

> Lee PLAN.md. Empecemos por el Día 1: crea la estructura de carpetas, el `.gitignore` y `requirements.txt`, e implementa `src/audio.py` y `src/transcripcion.py`. Prueba con un fragmento de 2 minutos de `data/videos/reunion_1` y muéstrame la transcripción.

## 11. Estado y decisiones (actualizado tras el Día 3)

### Hecho
- **Día 1:** estructura, `src/audio.py`, `src/transcripcion.py` (Whisper), prueba con fragmento de 2 min.
- **Día 2:** las 3 reuniones transcritas y guardadas; `src/texto.py` (corrección con glosario/variantes, limpieza de muletillas, oraciones); `src/nlp.py` con entidades y fechas; `tests/test_texto.py` (7 tests `unittest`: `python -m unittest`).
- **Día 3:** `extraer_palabras_clave()` y `extraer_temas()` (KeyBERT, `paraphrase-multilingual-MiniLM-L12-v2`).
- **Siguiente:** Día 4 (`src/resumen.py`). No iniciado; se espera confirmación.

### Decisiones clave
- **Whisper `medium` + glosario opcional.** `medium` baja el WER frente a `small`, y el `initial_prompt` con glosario lo baja más. Sin glosario no se usa prompt; el sistema no trae términos por defecto (ejemplo en `config/ejemplos/caso_reuniones/`).
- **Reparación automática de bucles y huecos** (activa por defecto): re-transcribe solo el tramo con problemas y registra el antes/después en `resultado["reparaciones"]`.
- **Texto fiel vs. texto limpio.** El *fiel* (nombres corregidos, con muletillas) es la transcripción que se entrega. El *limpio* (sin muletillas) se usa solo para análisis (entidades, fechas, palabras clave, resumen) y nunca se entrega como transcripción.
- **Entidades:** solo tokens PROPN (se recortan los extremos que no lo son), etiqueta más frecuente, fusión de variantes con rapidfuzz ≥ 90 (aunque la etiqueta difiera), glosario con prioridad (ORG).
- **Fechas:** absolutas (incluye el día del mes, "el 23") y relativas; se unifican variantes y se descartan las débiles ("el mes", "este año").
- **Palabras clave: sintagmas nominales de spaCy** como candidatas de KeyBERT (ngramas 1-2, MMR 0.5, stopwords en español), con filtros de ruido: mínimo 2 apariciones, sin palabras de discurso (gracias, momento, vez…), solo palabras del vocabulario español, PROPN o términos del glosario.

### Resultados medidos
**WER** (fragmento de 2 min de reunion_1, referencia manual `data/referencia/reunion_1_300s_120s_referencia.txt`):

| Configuración | WER | WER sin tildes |
|---|---|---|
| small | 15.3 % | 14.8 % |
| small + glosario | 13.5 % | 13.1 % |
| medium | 14.4 % | 14.0 % |
| **medium + glosario** | **12.2 %** | **11.8 %** |

Solo hay un fragmento de referencia (2 min), así que la diferencia entre configuraciones es orientativa.

**Transcripción completa (medium + glosario):**

| Reunión | Audio | Transcripción | Palabras | Reparaciones |
|---|---|---|---|---|
| 1 | 30.4 min | 277.7 s | 3657 | 2 |
| 2 | 23.0 min | 166.4 s | 2145 | 3 |
| 3 | 18.3 min | 122.3 s | 2297 | 2 |

**Correcciones sobre el texto ya transcrito** (reemplazos de palabra; todas fueron variantes de la sigla del glosario salvo lo indicado):

| Reunión | Variantes | Glosario |
|---|---|---|
| 1 | 5 | 10 (Perumeg/Perúmeg→PERUMEC, Asme→ASME) |
| 2 | 1 | 4 (→PERUMEC) |
| 3 | 0 | 4 (parter→partners, →PERUMEC) |

**Palabras clave, candidatas libres vs. sintagmas nominales** (mismas 3 reuniones): las libres dan pares verbo+sustantivo sin sentido ("ponente tendríamos", "presentaba presentar", "dijiste reunión"); los sintagmas dan sustantivos reales (ponente, comunicación, formulario, embajadores…). Se eligieron los sintagmas. Tras el filtro de ruido, ejemplo reunion_2: reunión, ponente, comunicación, coordinador, flyer, mañana, partners, marketing, posibilidad.

**Entidades** (top, con glosario): ver CSV de revisión abajo. Con glosario aparecen `partners` y suben ASME/PERUMEC, que spaCy solo no detecta bien. Solo hubo una fusión por similitud (Steve→Steven). Entre las 3 reuniones aparecen decenas de entidades, con errores de etiqueta de spaCy.

### Pendientes (del autor)
- **Llenar con s/n la columna `correcta`** de `data/referencia/entidades_reunion_1.csv` y `entidades_reunion_3.csv`; luego `evaluar_precision_entidades(csv)` (en `src/evaluacion.py`) da la precisión por tipo. No se versionan (`data/` está ignorado).
- **Fragmento de referencia de reunion_3:** transcribir a mano 2-3 min para un segundo WER (hoy solo hay de reunion_1).
- ROUGE: resumen de referencia a mano de un video (Día 6).

### Limitaciones conocidas (se suman a la sección 8)
- Las palabras clave son sobre todo palabras sueltas (spaCy da pocos bigramas), y aún se cuelan algunas ("hazme") y nombres mal etiquetados por spaCy (p. ej. "cybermeg").
- El NER de spaCy etiqueta mal algunos nombres (organizaciones como PER/LOC); la fusión y el glosario corrigen solo parte.
- Las variantes de una misma entidad con similitud < 90 (p. ej. "Cyber Mej" / "Cybermeg") no se unen.
- La limpieza de muletillas es conservadora: solo borra marcadores claramente aislados.
- El WER se midió sobre un solo fragmento de 2 min.
