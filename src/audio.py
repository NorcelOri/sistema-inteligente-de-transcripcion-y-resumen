"""Preparación del audio: extracción con ffmpeg, normalización y reducción de ruido.

Whisper trabaja con audio mono a 16 kHz, así que todo se convierte a ese formato.
La segmentación para el reconocimiento la hace Whisper internamente (ventanas de 30 s).
"""

import subprocess
from pathlib import Path

import noisereduce as nr
from pydub import AudioSegment
from scipy.io import wavfile

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_AUDIOS = RAIZ / "data" / "audios"

FORMATOS_VALIDOS = {".mp3", ".wav", ".m4a", ".mp4"}
FRECUENCIA = 16000  # Hz, la que espera Whisper


def verificar_formato(ruta) -> bool:
    """True si el archivo existe y tiene una extensión soportada."""
    ruta = Path(ruta)
    return ruta.is_file() and ruta.suffix.lower() in FORMATOS_VALIDOS


def duracion_segundos(ruta) -> float:
    """Duración del archivo (audio o video) usando ffprobe."""
    comando = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(ruta),
    ]
    salida = subprocess.run(comando, capture_output=True, text=True, check=True)
    return float(salida.stdout.strip())


def hash_archivo(ruta, n=10) -> str:
    """Primeros n caracteres del SHA-256 del contenido del archivo."""
    import hashlib
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()[:n]


def extraer_audio(ruta_entrada, carpeta_salida=CARPETA_AUDIOS,
                  inicio=None, duracion=None, nombre=None) -> Path:
    """Convierte el archivo de entrada a WAV mono 16 kHz.

    inicio y duracion (en segundos) son opcionales y permiten extraer
    solo un fragmento, útil para pruebas rápidas y para el WER.
    Si el WAV ya existe, se reutiliza.
    """
    ruta_entrada = Path(ruta_entrada)
    if not verificar_formato(ruta_entrada):
        raise ValueError(f"Formato no soportado o archivo inexistente: {ruta_entrada}")

    # Nombre del WAV: si es un fragmento, se indica en el nombre
    nombre = nombre or ruta_entrada.stem
    if inicio is not None or duracion is not None:
        nombre += f"_{int(inicio or 0)}s_{int(duracion or 0)}s"
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_wav = carpeta_salida / f"{nombre}.wav"

    if ruta_wav.exists():
        return ruta_wav

    comando = ["ffmpeg", "-y", "-loglevel", "error"]
    if inicio is not None:
        comando += ["-ss", str(inicio)]      # antes de -i: salto rápido
    comando += ["-i", str(ruta_entrada)]
    if duracion is not None:
        comando += ["-t", str(duracion)]
    comando += [
        "-vn",                    # descartar el video
        "-ac", "1",               # mono
        "-ar", str(FRECUENCIA),   # 16 kHz
        "-c:a", "pcm_s16le",      # WAV de 16 bits
        str(ruta_wav),
    ]
    subprocess.run(comando, check=True)
    return ruta_wav


def normalizar_volumen(ruta_wav, dbfs_objetivo=-20.0) -> Path:
    """Ajusta el volumen medio del audio a dbfs_objetivo. Devuelve un WAV nuevo."""
    ruta_wav = Path(ruta_wav)
    audio = AudioSegment.from_wav(ruta_wav)
    audio = audio.apply_gain(dbfs_objetivo - audio.dBFS)
    ruta_salida = ruta_wav.with_name(f"{ruta_wav.stem}_norm.wav")
    audio.export(ruta_salida, format="wav")
    return ruta_salida


def reducir_ruido(ruta_wav) -> Path:
    """Aplica reducción de ruido espectral (noisereduce). Devuelve un WAV nuevo.

    Es opcional: sirve para comparar el WER con y sin reducción de ruido.
    """
    ruta_wav = Path(ruta_wav)
    frecuencia, datos = wavfile.read(ruta_wav)
    limpio = nr.reduce_noise(y=datos, sr=frecuencia)
    ruta_salida = ruta_wav.with_name(f"{ruta_wav.stem}_nr.wav")
    wavfile.write(ruta_salida, frecuencia, limpio.astype(datos.dtype))
    return ruta_salida
