"""Métricas de evaluación: WER (jiwer)."""

import re
import unicodedata

import jiwer


def normalizar(texto: str, quitar_tildes=False) -> str:
    """Minúsculas, sin puntuación y con espacios colapsados.

    Con quitar_tildes=True también se eliminan los acentos, pero se conserva la ñ.
    """
    texto = texto.lower()
    if quitar_tildes:
        # Proteger la ñ: descomponer, quitar marcas combinantes, recomponer
        texto = texto.replace("ñ", "\0")
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        texto = texto.replace("\0", "ñ")
    texto = re.sub(r"[^\w\s]|_", " ", texto)  # puntuación -> espacio
    return re.sub(r"\s+", " ", texto).strip()


def wer(referencia: str, hipotesis: str) -> dict:
    """WER normal (con tildes) y WER sin tildes, como fracciones (0.10 = 10 %)."""
    resultado = {}
    for clave, sin_tildes in (("wer", False), ("wer_sin_tildes", True)):
        ref = normalizar(referencia, sin_tildes)
        hip = normalizar(hipotesis, sin_tildes)
        resultado[clave] = jiwer.wer(ref, hip)
    return resultado


def errores(referencia: str, hipotesis: str, quitar_tildes=False) -> list:
    """Lista de errores palabra a palabra: (tipo, referencia, hipótesis).

    tipo: "sustitución", "inserción" o "omisión".
    """
    ref = normalizar(referencia, quitar_tildes)
    hip = normalizar(hipotesis, quitar_tildes)
    salida = jiwer.process_words(ref, hip)
    palabras_ref, palabras_hip = ref.split(), hip.split()
    lista = []
    for alineacion in salida.alignments[0]:
        r = palabras_ref[alineacion.ref_start_idx:alineacion.ref_end_idx]
        h = palabras_hip[alineacion.hyp_start_idx:alineacion.hyp_end_idx]
        if alineacion.type == "substitute":
            lista += [("sustitución", a, b) for a, b in zip(r, h)]
        elif alineacion.type == "delete":
            lista += [("omisión", a, "") for a in r]
        elif alineacion.type == "insert":
            lista += [("inserción", "", b) for b in h]
    return lista
