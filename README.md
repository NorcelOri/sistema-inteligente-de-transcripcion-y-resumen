# Sistema inteligente de transcripción y resumen automático de audios

Prototipo (en desarrollo) del proyecto de la Unidad 1 del curso de Procesamiento de Lenguaje Natural, Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional del Altiplano Puno.

**Autor:** Cuyo Zamata Yimmy Yeyson · **Docente:** Allca Mamani Ubaldo

## Qué hace

A partir de un audio o video en español (.mp3, .wav, .m4a, .mp4), el sistema entrega:

- Transcripción completa (Whisper `medium`), con reparación automática de bucles y huecos.
- Corrección opcional de nombres propios mediante un glosario del usuario.
- Palabras clave y temas (KeyBERT + spaCy).
- Entidades (personas, lugares, organizaciones) y fechas.
- Resumen automático (TextRank por defecto; mT5 con verificación como opción experimental).
- Archivos descargables: transcripción `.txt` y resumen `.pdf`, desde una interfaz web (Gradio).

## Requisitos

- Python 3.12, ffmpeg y una GPU NVIDIA con CUDA (probado con RTX 5070, CUDA 12.8).
- Windows (el PDF usa la fuente Arial de `C:\Windows\Fonts`).

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m spacy download es_core_news_md
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Uso

```bash
python app.py
```

Abre http://127.0.0.1:7860, sube un audio o video, escribe (opcional) los términos propios del audio y pulsa **Procesar**.

## Estructura

- `app.py`: interfaz Gradio.
- `src/`: módulos (audio, transcripción, texto, NLP, resumen, salidas, evaluación y pipeline).
- `tests/`: pruebas automáticas.
- `PLAN.md`: estado, decisiones y resultados medidos.

## Resultados principales (preliminares)

- WER en un fragmento de prueba: 12.2 % con Whisper `medium` + prompt (8.7 % con corrección de nombres ajustada sobre ese fragmento).
- Resumen: TextRank supera a mT5 en ROUGE y no inventa contenido; mT5 genera datos que no están en el audio.
- Precisión de entidades: entre 0.42 y 0.60 (muestras pequeñas).

## Limitaciones

Sin diarización de hablantes, optimizado solo para español, resumen extractivo con estilo oral, y evaluación con muestras pequeñas. Se trabajará en la Unidad 2.

## Privacidad

Los audios, transcripciones y referencias de prueba corresponden a reuniones reales con consentimiento de los participantes y **no se incluyen en este repositorio**.
