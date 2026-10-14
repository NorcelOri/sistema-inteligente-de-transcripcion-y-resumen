"""Información relevante del texto: entidades (spaCy) y fechas (regex).

Se aplica sobre el texto limpio. El NER sobre habla transcrita es ruidoso:
se filtran entidades de un carácter, sin mayúscula o sin letras.
"""

import re
import unicodedata
from collections import Counter

from src.texto import _cargar_spacy, _es_palabra_comun

TIPOS_ENTIDAD = ("PER", "LOC", "ORG")

# Artículos y demostrativos al inicio ("el Perú", "esa Cyber Mej") para unir variantes
_ARTICULO = re.compile(r"^(?:el|la|los|las|un|una|este|esta|ese|esa|aquel|aquella)\s+", re.IGNORECASE)


def _clave(texto: str) -> str:
    """Clave para unir variantes de la misma entidad (sin tildes ni mayúsculas)."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip(" .,;:")


def _es_valida(texto: str) -> bool:
    """Descarta entidades de un carácter, sin mayúscula, o que son una sola
    palabra común del español capitalizada ("Bueno", "Claro", "Mañana": la
    mayúscula es solo de inicio de oración)."""
    letras = [c for c in texto if c.isalpha()]
    if len(letras) < 2 or not any(c.isupper() for c in letras):
        return False
    una_palabra = len(texto.split()) == 1
    return not (una_palabra and not texto.isupper() and _es_palabra_comun(texto))


def extraer_entidades(texto: str, tipos=TIPOS_ENTIDAD, minimo=1) -> dict:
    """Entidades por tipo: {"PER": [(entidad, frecuencia), ...], ...}.

    Une variantes que solo difieren en mayúsculas/tildes y ordena por
    frecuencia (de mayor a menor). `minimo`: frecuencia mínima para incluirla.
    """
    doc = _cargar_spacy()(texto)
    conteo = {t: Counter() for t in tipos}
    formas = {t: {} for t in tipos}   # clave -> Counter de formas escritas
    for ent in doc.ents:
        if ent.label_ in conteo:
            nombre = _ARTICULO.sub("", ent.text.strip(" .,;:¿?¡!"))
            if _es_valida(nombre):
                clave = _clave(nombre)
                conteo[ent.label_][clave] += 1
                formas[ent.label_].setdefault(clave, Counter())[nombre] += 1
    salida = {}
    for tipo in tipos:
        lista = [(formas[tipo][c].most_common(1)[0][0], n)
                 for c, n in conteo[tipo].items() if n >= minimo]
        salida[tipo] = sorted(lista, key=lambda e: (-e[1], e[0].lower()))
    return salida


# ---------------------------------------------------------------------------
# Fechas
# ---------------------------------------------------------------------------

_MESES = (r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
          r"setiembre|septiembre|octubre|noviembre|diciembre")
_DIAS = r"lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo"
_NUM = r"(?:[0-2]?\d|3[01]|primero)"

_PATRONES_FECHA = [
    # 12/10/2026, 12-10-26, 12/10
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    # 15 de octubre (de 2026) / 1 de marzo
    rf"\b{_NUM}\s+de\s+(?:{_MESES})(?:\s+(?:de|del)\s+\d{{4}})?\b",
    # octubre de 2026 / en marzo
    rf"\b(?:{_MESES})\s+(?:de|del)\s+\d{{4}}\b",
    # el lunes, el próximo viernes, este sábado, el lunes 15
    rf"\b(?:(?:el|este|esta|del|al|para el|hasta el)\s+)?(?:(?:pr[oó]ximo|pasado|siguiente)\s+)?(?:{_DIAS})\b(?:\s+{_NUM}\b)?",
    # relativas
    r"\b(?:pasado\s+ma[ñn]ana|anteayer|antes\s+de\s+ayer|ayer|hoy|anoche|esta\s+noche)\b",
    r"(?<!por la )(?<!en la )(?<!de la )(?<!a la )(?<!la )\bma[ñn]ana\b",
    r"\b(?:la|esta|el|este|la\s+pr[oó]xima|la\s+siguiente|la\s+[uú]ltima)\s+(?:semana|mes|a[ñn]o)(?:\s+(?:pasada?|pr[oó]xim[oa]|que\s+viene))?\b",
    r"\b(?:fin\s+de\s+semana|fines\s+de\s+semana)\b",
    # el 23, para el 30, hasta el 20 (día suelto): no seguido de unidades o de otra cifra
    rf"\b(?:el|del|al|para el|hasta el)\s+{_NUM}\b(?![/-]\d)(?!\s*(?:de\b|%|por\s*ciento|soles|d[oó]lares|personas|horas|minutos|primeros|inscritos|:|\d))",
    # año suelto
    r"\b(?:en\s+)?(?:19|20)\d{2}\b",
]
_FECHAS = re.compile("|".join(f"(?:{p})" for p in _PATRONES_FECHA), re.IGNORECASE)


def extraer_fechas(texto: str) -> list:
    """Expresiones de fecha: [(expresión, frecuencia), ...] de mayor a menor.

    Une variantes de mayúsculas y espacios. Incluye fechas completas
    ("20 de marzo"), días de la semana, relativas ("mañana", "la próxima
    semana") y días sueltos ("el 23").
    """
    conteo = Counter()
    for m in _FECHAS.finditer(texto):
        expresion = re.sub(r"\s+", " ", m.group(0).strip().lower())
        conteo[expresion] += 1
    return sorted(conteo.items(), key=lambda e: (-e[1], e[0]))
