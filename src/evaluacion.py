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


def evaluar_precision_entidades(ruta_csv) -> dict:
    """Precisión de las entidades por tipo, desde el CSV revisado a mano.

    El CSV tiene las columnas entidad, tipo, frecuencia, correcta (s/n).
    Precisión = entidades marcadas "s" / entidades revisadas, por tipo y en total.
    Si falta alguna marca, lanza ValueError (el CSV aún no está completo).
    """
    import csv
    with open(ruta_csv, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    marcas = {"s": True, "si": True, "sí": True, "n": False, "no": False}
    vacias = [f["entidad"] for f in filas if f["correcta"].strip().lower() not in marcas]
    if vacias:
        raise ValueError(f"Faltan o son inválidas {len(vacias)} marcas (usa s/n): {vacias[:5]}...")
    resultado = {}
    for tipo in sorted({f["tipo"] for f in filas}) + ["TOTAL"]:
        grupo = [marcas[f["correcta"].strip().lower()] for f in filas if tipo in (f["tipo"], "TOTAL")]
        resultado[tipo] = {"revisadas": len(grupo), "correctas": sum(grupo),
                           "precision": round(sum(grupo) / len(grupo), 3)}
    return resultado
