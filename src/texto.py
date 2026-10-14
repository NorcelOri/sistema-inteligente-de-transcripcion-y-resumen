"""Procesamiento del texto transcrito.

Por ahora: corrección de nombres propios mal transcritos, en dos capas:
  1. Glosario (opcional): corrección aproximada con rapidfuzz.
  2. Variantes (opcional): diccionario "variante => correcta" de errores ya vistos.
(Limpieza de muletillas y segmentación en oraciones: siguiente etapa.)
"""

import re
import unicodedata

from rapidfuzz import fuzz

# Umbral de similitud (0-100) según el largo del término. Conservador: los
# términos cortos (siglas) solo se corrigen si coinciden salvo mayúsculas.
UMBRAL_LARGO = 85      # términos de 6 o más letras
UMBRAL_CORTO = 100     # términos de 5 o menos letras: coincidencia exacta
RANGO_COMUN = 10000    # palabras entre las 10 000 más frecuentes = comunes
_PALABRA = re.compile(r"\w+", re.UNICODE)

_spacy = {}


def _sin_tildes(s: str) -> str:
    s = s.lower().replace("ñ", "\0")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return s.replace("\0", "ñ")


def _es_palabra_comun(palabra: str) -> bool:
    """True si es una palabra frecuente del español.

    Usa el vocabulario de spaCy, cuyos vectores están ordenados por frecuencia:
    las primeras RANGO_COMUN filas son las palabras más usadas (el resto incluye
    ruido como 'mej').
    """
    if "nlp" not in _spacy:
        try:
            import spacy
            _spacy["nlp"] = spacy.load("es_core_news_md", disable=["parser", "ner"])
        except Exception:
            _spacy["nlp"] = None
    nlp = _spacy["nlp"]
    if not nlp:
        return False
    fila = nlp.vocab.vectors.key2row.get(nlp.vocab.strings[palabra.lower()])
    return fila is not None and fila < RANGO_COMUN


def _umbral(termino: str) -> int:
    return UMBRAL_LARGO if len(_sin_tildes(termino).replace(" ", "")) >= 6 else UMBRAL_CORTO


def _candidatos(palabras, termino):
    """Ventanas de palabras consecutivas del texto que podrían ser el término.

    Tamaños: igual al del término, uno más y uno menos (por si Whisper partió
    o juntó palabras, como 'perú mej' por 'PERUMEC').
    """
    k = len(termino.split())
    for n in sorted({max(1, k - 1), k, k + 1}):
        for i in range(len(palabras) - n + 1):
            yield i, n


def corregir_glosario(texto, glosario=None) -> str:
    """Corrige nombres del glosario que Whisper escribió parecido pero mal.

    Coincidencia aproximada por palabra o frase (rapidfuzz). Conservador:
    - misma letra inicial,
    - umbral alto; siglas y términos cortos solo corrigen mayúsculas,
    - no toca palabras comunes del español (10 000 más frecuentes de spaCy): una ventana
      con alguna palabra común solo se corrige si coincide exacta.
    Con glosario vacío o None no hace nada.
    """
    if not glosario:
        return texto
    palabras = [(m.start(), m.end(), m.group()) for m in _PALABRA.finditer(texto)]
    if not palabras:
        return texto
    normales = [_sin_tildes(p[2]) for p in palabras]

    hallazgos = []  # (puntaje, i, n, termino)
    for termino in glosario:
        objetivo = _sin_tildes(termino).replace(" ", "")
        umbral = _umbral(termino)
        for i, n in _candidatos(normales, termino):
            ventana = "".join(normales[i:i + n])
            if ventana[0] != objetivo[0]:
                continue
            puntaje = fuzz.ratio(ventana, objetivo)
            if puntaje < umbral:
                continue
            # Si incluye alguna palabra común y no es el término exacto, no se toca
            # ('perú mej' sí pasa; 'perú me' o 'parter es' no: se comerían 'me'/'es')
            if puntaje < 100 and any(_es_palabra_comun(p[2]) for p in palabras[i:i + n]):
                continue
            hallazgos.append((puntaje, i, n, termino))

    # Mejor puntaje primero; los que se solapan con uno ya aceptado se descartan
    ocupadas, aceptados = set(), []
    for puntaje, i, n, termino in sorted(hallazgos, key=lambda h: (-h[0], -h[2])):
        rango = set(range(i, i + n))
        if rango & ocupadas:
            continue
        ocupadas |= rango
        aceptados.append((i, n, termino))

    # Reemplazar de atrás hacia adelante para no mover las posiciones
    for i, n, termino in sorted(aceptados, reverse=True):
        inicio, fin = palabras[i][0], palabras[i + n - 1][1]
        original = texto[inicio:fin]
        nuevo = termino
        if not termino.isupper() and original[0].isupper():
            nuevo = termino[0].upper() + termino[1:]
        texto = texto[:inicio] + nuevo + texto[fin:]
    return texto


def corregir_variantes(texto, variantes=None) -> str:
    """Reemplaza variantes conocidas (palabras completas, sin distinguir mayúsculas)."""
    if not variantes:
        return texto
    claves = sorted(variantes, key=len, reverse=True)  # las más largas primero
    patron = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(c).replace(r"\ ", r"\s+") for c in claves) + r")(?!\w)",
        re.IGNORECASE,
    )

    def reemplazo(m):
        correcta = variantes[re.sub(r"\s+", " ", m.group(1).lower())]
        if not correcta.isupper() and m.group(1)[0].isupper():
            correcta = correcta[0].upper() + correcta[1:]
        return correcta

    return patron.sub(reemplazo, texto)


def corregir_nombres(texto, glosario=None, variantes=None) -> str:
    """Corrige nombres propios: primero variantes conocidas, luego el glosario.

    Ambos son opcionales; sin ninguno devuelve el texto igual.
    """
    return corregir_glosario(corregir_variantes(texto, variantes), glosario)
