"""Transcripción de audio con Whisper (modelo preentrenado de OpenAI).

El modelo se carga una sola vez y se reutiliza. Cada transcripción se guarda
en data/transcripciones/ para no repetir el trabajo de GPU.

Incluye reparación automática de bucles (frases repetidas) y de huecos largos
sin texto dentro de habla.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import whisper
from rapidfuzz import fuzz

from src.evaluacion import normalizar

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


def construir_prompt(glosario=None):
    """initial_prompt para Whisper a partir del glosario del usuario.

    Sin glosario (None o lista vacía) devuelve None: no se usa prompt.
    """
    return "Vocabulario: " + ", ".join(glosario) + "." if glosario else None


def _rutas_cache(ruta_wav, modelo, prompt=None):
    """Rutas del .txt (texto plano) y .json (texto + segmentos) de una transcripción."""
    # La clave incluye el contenido (el nombre del WAV lleva su hash), el modelo y el glosario
    sufijo = "_prompt_" + hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8] if prompt else ""
    base = CARPETA_TRANSCRIPCIONES / f"{Path(ruta_wav).stem}_{modelo}{sufijo}"
    return base.with_suffix(".txt"), base.with_suffix(".json")


def transcribir(ruta_wav, idioma="es", modelo="small", usar_cache=True,
                glosario=None, reparar=True) -> dict:
    """Transcribe un WAV y devuelve:

    {"texto": str, "segmentos": [{"inicio", "fin", "texto"}, ...],
     "idioma": str, "modelo": str, "reparaciones": [...]}

    glosario: lista opcional de términos (nombres propios, siglas) que orienta
    la escritura de Whisper mediante initial_prompt. Vacío por defecto.
    reparar: detecta bucles y huecos largos y re-transcribe esos tramos.
    """
    initial_prompt = construir_prompt(glosario)
    ruta_txt, ruta_json = _rutas_cache(ruta_wav, modelo, initial_prompt)
    ruta_rep = ruta_json.with_name(ruta_json.stem + "_rep.json")

    # Si ya se transcribió antes con este modelo, reutilizar
    if usar_cache and reparar and ruta_rep.exists():
        return json.loads(ruta_rep.read_text(encoding="utf-8"))
    if usar_cache and ruta_json.exists():
        resultado = json.loads(ruta_json.read_text(encoding="utf-8"))
        return _reparar_y_guardar(resultado, ruta_wav, ruta_rep, initial_prompt) if reparar else resultado

    modelo_whisper = cargar_whisper(modelo)
    en_gpu = modelo_whisper.device.type == "cuda"
    salida = modelo_whisper.transcribe(str(ruta_wav), language=idioma, fp16=en_gpu,
                                       initial_prompt=initial_prompt)

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
    if reparar:
        return _reparar_y_guardar(resultado, ruta_wav, ruta_rep, initial_prompt)
    return resultado


def _reparar_y_guardar(resultado, ruta_wav, ruta_rep, initial_prompt):
    """Repara la transcripción y guarda el resultado en disco."""
    resultado = reparar_transcripcion(resultado, ruta_wav, initial_prompt)
    ruta_rep.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    ruta_rep.with_suffix(".txt").write_text(resultado["texto"], encoding="utf-8")
    return resultado


# ---------------------------------------------------------------------------
# Reparación automática: bucles y huecos
#
# Whisper a veces se "atasca" repitiendo una frase (bucle) o deja sin texto un
# tramo largo donde sí hay voz. Ambos fallos se detectan en los segmentos y se
# corrigen re-transcribiendo solo ese tramo con condition_on_previous_text=False
# (sin arrastrar el texto anterior, que es lo que alimenta el bucle).
# ---------------------------------------------------------------------------

MIN_REPETICIONES = 3        # frase idéntica repetida 3+ veces seguidas = bucle
MIN_PALABRAS_REP2 = 3       # con solo 2 repeticiones, la frase debe tener 3+ palabras
MIN_HUECO = 8.0             # segundos sin texto para sospechar
MIN_ACTIVIDAD = 0.5         # fracción mínima de audio con voz dentro del hueco
MARGEN = 2.0                # segundos de contexto alrededor del tramo
FRAME = 480                 # 30 ms a 16 kHz, para medir energía


def detectar_bucles(segmentos) -> list:
    """Tramos (inicio, fin) de segmentos consecutivos con el mismo texto."""
    tramos, i = [], 0
    while i < len(segmentos):
        clave = normalizar(segmentos[i]["texto"])
        j = i + 1
        while j < len(segmentos) and fuzz.ratio(clave, normalizar(segmentos[j]["texto"])) >= 95:
            j += 1
        repeticiones = j - i
        if clave and (repeticiones >= MIN_REPETICIONES or
                      (repeticiones == 2 and len(clave.split()) >= MIN_PALABRAS_REP2)):
            tramos.append((segmentos[i]["inicio"], segmentos[j - 1]["fin"]))
        i = j
    return tramos


def _actividad(audio, inicio, fin) -> float:
    """Fracción de fotogramas del tramo con energía de voz (frente al piso de ruido)."""
    n = len(audio) // FRAME
    energia = np.sqrt((audio[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    umbral = max(np.percentile(energia, 10) * 3, 0.005)
    fps = 16000 / FRAME
    tramo = energia[int(inicio * fps):int(fin * fps)]
    return float((tramo > umbral).mean()) if len(tramo) else 0.0


def detectar_huecos(segmentos, audio) -> list:
    """Tramos (inicio, fin) largos sin texto pero con voz según la energía."""
    huecos = []
    for a, b in zip(segmentos, segmentos[1:]):
        if b["inicio"] - a["fin"] >= MIN_HUECO and _actividad(audio, a["fin"], b["inicio"]) >= MIN_ACTIVIDAD:
            huecos.append((a["fin"], b["inicio"]))
    return huecos


def _unir_tramos(tramos, margen=MARGEN) -> list:
    """Ordena y junta los tramos que se tocan o están a menos de `margen`."""
    unidos = []
    for ini, fin in sorted(tramos):
        if unidos and ini - unidos[-1][1] <= margen:
            unidos[-1] = (unidos[-1][0], max(unidos[-1][1], fin))
        else:
            unidos.append((ini, fin))
    return unidos


def _retranscribir(audio, inicio, fin, idioma, modelo, initial_prompt, forzar=False) -> list:
    """Transcribe audio[inicio:fin] sin condicionar al texto previo.

    forzar=True desactiva los descartes de Whisper por "probablemente silencio"
    (no_speech_threshold) y por baja confianza (logprob_threshold), que son los
    que dejan huecos donde sí hay voz.
    """
    modelo_whisper = cargar_whisper(modelo)
    trozo = audio[int(inicio * 16000):int(fin * 16000)]
    opciones = {"no_speech_threshold": None, "logprob_threshold": None} if forzar else {}
    torch.manual_seed(0)  # el muestreo de respaldo de Whisper es aleatorio: fijarlo
    salida = modelo_whisper.transcribe(
        trozo, language=idioma, fp16=modelo_whisper.device.type == "cuda",
        condition_on_previous_text=False, initial_prompt=initial_prompt, **opciones)
    return [{"inicio": round(inicio + s["start"], 2), "fin": round(inicio + s["end"], 2),
             "texto": s["text"].strip()} for s in salida["segments"] if s["text"].strip()]


def _problemas_en_tramo(nuevos, audio, a, b) -> int:
    """Cuenta bucles y huecos con voz dentro del tramo re-transcrito (bordes incluidos)."""
    centinela_ini = {"inicio": a, "fin": a, "texto": ""}
    centinela_fin = {"inicio": b, "fin": b, "texto": ""}
    con_bordes = [centinela_ini] + nuevos + [centinela_fin]
    return len(detectar_bucles(nuevos)) + len(detectar_huecos(con_bordes, audio))


def _quitar_repetidos_en_bordes(segmentos, nuevos, vecinos=3):
    """Quita los segmentos conservados junto al tramo nuevo cuyo texto ya está en él.

    Los tiempos de Whisper no son exactos: al re-transcribir un tramo, el
    audio puede traer frases que el segmento vecino ya tenía.
    """
    if not nuevos:
        return segmentos
    nuevo = normalizar(" ".join(s["texto"] for s in nuevos))
    ids = {id(s) for s in nuevos}
    pos = [i for i, s in enumerate(segmentos) if id(s) in ids]
    cerca = set(range(max(0, pos[0] - vecinos), pos[0])) | set(range(pos[-1] + 1, pos[-1] + 1 + vecinos))
    quitar = set()
    for i in cerca:
        if i < len(segmentos):
            texto = normalizar(segmentos[i]["texto"])
            if len(texto.split()) >= 4 and fuzz.partial_ratio(texto, nuevo) >= 85:
                quitar.add(i)
    return [s for i, s in enumerate(segmentos) if i not in quitar]


def reparar_transcripcion(resultado, ruta_wav, initial_prompt=None) -> dict:
    """Detecta bucles y huecos y re-transcribe esos tramos.

    Devuelve una copia del resultado con segmentos corregidos y la lista
    "reparaciones": [{"tipo", "inicio", "fin", "antes", "despues"}, ...].
    """
    segmentos = resultado["segmentos"]
    audio = whisper.load_audio(str(ruta_wav))
    marcados = ([("bucle", t) for t in detectar_bucles(segmentos)] +
                [("hueco", t) for t in detectar_huecos(segmentos, audio)])
    if not marcados:
        return {**resultado, "reparaciones": []}

    reparaciones = []
    for ini, fin in _unir_tramos([t for _, t in marcados]):
        tipos = sorted({tipo for tipo, (a, b) in marcados if a <= fin and b >= ini})
        # Ampliar con margen y a segmentos completos
        ini_m, fin_m = max(0.0, ini - MARGEN), min(len(audio) / 16000, fin + MARGEN)
        afectados = [s for s in segmentos if s["fin"] > ini_m and s["inicio"] < fin_m]
        a = min([ini_m] + [s["inicio"] for s in afectados])
        b = max([fin_m] + [s["fin"] for s in afectados])
        # Primer intento normal; si el tramo sigue con bucle o hueco, segundo intento forzado
        nuevos, mejor = None, None
        for forzar in (False, True):
            intento = _retranscribir(audio, a, b, resultado["idioma"], resultado["modelo"],
                                     initial_prompt, forzar)
            problemas = _problemas_en_tramo(intento, audio, a, b)
            if mejor is None or problemas < mejor:
                nuevos, mejor = intento, problemas
            if problemas == 0:
                break
        segmentos = [s for s in segmentos if s not in afectados] + nuevos
        segmentos.sort(key=lambda s: s["inicio"])
        segmentos = _quitar_repetidos_en_bordes(segmentos, nuevos)
        reparaciones.append({"tipo": "+".join(tipos), "inicio": a, "fin": b,
                             "problemas_restantes": mejor,
                             "antes": [s["texto"] for s in afectados],
                             "despues": [s["texto"] for s in nuevos]})

    texto = " ".join(s["texto"] for s in segmentos)
    return {**resultado, "texto": texto, "segmentos": segmentos, "reparaciones": reparaciones}
