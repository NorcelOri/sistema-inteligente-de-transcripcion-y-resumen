"""Métricas de evaluación: WER (jiwer)."""

import re
import unicodedata
from pathlib import Path

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


class _Tokenizador:
    """Tokenizador para rouge-score: minúsculas, sin puntuación, conserva tildes y ñ.

    El de rouge-score por defecto borra todo lo que no sea a-z y 0-9 (rompe
    "reunión" en "reuni n"). Se reutiliza `normalizar` (con o sin tildes).
    """

    def __init__(self, quitar_tildes=False):
        self.quitar_tildes = quitar_tildes

    def tokenize(self, texto):
        return normalizar(texto, self.quitar_tildes).split()


def rouge(referencia: str, resumen: str) -> dict:
    """ROUGE-1, ROUGE-2 y ROUGE-L (precisión, recall y F1), con y sin tildes.

    Devuelve {"con_tildes": {"rouge1": {"p", "r", "f"}, ...}, "sin_tildes": {...}}.
    """
    from rouge_score import rouge_scorer
    resultado = {}
    for clave, sin_tildes in (("con_tildes", False), ("sin_tildes", True)):
        evaluador = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                             tokenizer=_Tokenizador(sin_tildes))
        puntajes = evaluador.score(referencia, resumen)
        resultado[clave] = {k: {"p": round(v.precision, 3), "r": round(v.recall, 3),
                                "f": round(v.fmeasure, 3)} for k, v in puntajes.items()}
    return resultado


def evaluar_precision_entidades(ruta_csv, min_frecuencia=1) -> dict:
    """Precisión de las entidades por tipo, desde el CSV revisado a mano.

    El CSV tiene las columnas entidad, tipo, frecuencia, correcta (s/n).
    Precisión = entidades marcadas "s" / entidades revisadas, por tipo y en total.
    Solo cuentan las entidades con frecuencia >= min_frecuencia.
    Si falta alguna marca, lanza ValueError (el CSV aún no está completo).
    """
    import csv
    # Excel en español guarda con ";" y a veces en cp1252: se detectan ambos
    crudo = Path(ruta_csv).read_bytes()
    try:
        contenido = crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        contenido = crudo.decode("cp1252")
    delimitador = ";" if contenido.splitlines()[0].count(";") > contenido.splitlines()[0].count(",") else ","
    filas = [f for f in csv.DictReader(contenido.splitlines(), delimiter=delimitador)
             if (f.get("entidad") or "").strip() and int(f["frecuencia"]) >= min_frecuencia]
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
