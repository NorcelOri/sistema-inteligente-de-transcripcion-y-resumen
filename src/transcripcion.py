"""Transcripción de audio con Whisper (modelo preentrenado de OpenAI).

El modelo se carga una sola vez y se reutiliza. Cada transcripción se guarda
en data/transcripciones/ para no repetir el trabajo de GPU.
"""

import json
from pathlib import Path

import torch
import whisper

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_TRANSCRIPCIONES = RAIZ / "data" / "transcripciones"

# Cache del modelo cargado: {"nombre": ..., "modelo": ...}
_cache = {}


def cargar_whisper(modelo="small", device=None):
    """Carga el modelo Whisper (solo la primera vez) y lo devuelve."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _cache.get("nombre") != (modelo, device):
        _cache["modelo"] = whisper.load_model(modelo, device=device)
        _cache["nombre"] = (modelo, device)
    return _cache["modelo"]


def _rutas_cache(ruta_wav, modelo):
    """Rutas del .txt (texto plano) y .json (texto + segmentos) de una transcripción."""
    base = CARPETA_TRANSCRIPCIONES / f"{Path(ruta_wav).stem}_{modelo}"
    return base.with_suffix(".txt"), base.with_suffix(".json")


def transcribir(ruta_wav, idioma="es", modelo="small", usar_cache=True) -> dict:
    """Transcribe un WAV y devuelve:

    {"texto": str, "segmentos": [{"inicio", "fin", "texto"}, ...],
     "idioma": str, "modelo": str}
    """
    ruta_txt, ruta_json = _rutas_cache(ruta_wav, modelo)

    # Si ya se transcribió antes con este modelo, reutilizar
    if usar_cache and ruta_json.exists():
        return json.loads(ruta_json.read_text(encoding="utf-8"))

    modelo_whisper = cargar_whisper(modelo)
    en_gpu = modelo_whisper.device.type == "cuda"
    salida = modelo_whisper.transcribe(str(ruta_wav), language=idioma, fp16=en_gpu)

    resultado = {
        "texto": salida["text"].strip(),
        "segmentos": [
            {"inicio": round(s["start"], 2), "fin": round(s["end"], 2),
             "texto": s["text"].strip()}
            for s in salida["segments"]
        ],
        "idioma": idioma,
        "modelo": modelo,
    }

    # Guardar en disco apenas se genera
    CARPETA_TRANSCRIPCIONES.mkdir(parents=True, exist_ok=True)
    ruta_txt.write_text(resultado["texto"], encoding="utf-8")
    ruta_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado
