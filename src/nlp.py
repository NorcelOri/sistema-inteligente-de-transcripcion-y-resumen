"""Información relevante del texto: entidades (spaCy) y fechas (regex).

Se aplica sobre el texto limpio. El NER sobre habla transcrita es ruidoso:
se filtran entidades de un carácter, sin mayúscula o sin letras.
"""

import re
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

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


def _recortar_propn(ent) -> str:
    """Texto de la entidad sin los extremos que no son PROPN ("Gracias Steven"
    -> "Steven"). Los tokens internos ("Universidad de Puno") se conservan.
    Devuelve "" si ningún token es PROPN."""
    tokens = [t for t in ent if not t.is_punct]
    while tokens and tokens[0].pos_ != "PROPN":
        tokens.pop(0)
    while tokens and tokens[-1].pos_ != "PROPN":
        tokens.pop()
    return " ".join(t.text for t in tokens)


def _unificar(candidatas, umbral):
    """Une variantes parecidas (rapidfuzz >= umbral), aunque tengan etiquetas distintas.

    candidatas: {clave: {"n", "etiqueta", "formas"}}. Se procesan de más a menos
    frecuentes: la variante más frecuente da la forma canónica y la etiqueta.
    Devuelve (canonicas, fusiones) con fusiones = [(variante, etiqueta, canónica, etiqueta, puntaje)].
    """
    canonicas, fusiones = [], []
    for clave, c in sorted(candidatas.items(), key=lambda kv: -kv[1]["n"]):
        for canon in canonicas:
            puntaje = fuzz.ratio(clave, canon["clave"])
            if puntaje >= umbral:
                canon["n"] += c["n"]
                fusiones.append((c["formas"].most_common(1)[0][0], c["etiqueta"],
                                 canon["formas"].most_common(1)[0][0], canon["etiqueta"], round(puntaje)))
                break
        else:
            canonicas.append({"clave": clave, **c})
    return canonicas, fusiones


def extraer_entidades(texto: str, tipos=TIPOS_ENTIDAD, minimo=1, glosario=None,
                      umbral_fusion=90, registro=None) -> dict:
    """Entidades por tipo: {"PER": [(entidad, frecuencia), ...], ...}.

    - Solo tokens PROPN (spaCy); se recortan los extremos que no lo son.
    - La etiqueta de cada entidad es la más frecuente entre sus menciones.
    - Une variantes parecidas (rapidfuzz, umbral 90, aunque la etiqueta difiera;
      la variante más frecuente da la forma y la etiqueta).
    - Los términos del glosario tienen prioridad y se etiquetan ORG.
    `minimo`: frecuencia mínima. `registro` (dict opcional) se llena con
    "fusiones" y "eliminadas" (por filtro) para revisión.
    """
    registro = registro if registro is not None else {}
    registro["fusiones"], registro["eliminadas"] = [], {"sin PROPN": [], "no válida": []}
    doc = _cargar_spacy()(texto)

    # 1. Menciones válidas, unidas por clave (sin tildes ni mayúsculas)
    candidatas = {}
    for ent in doc.ents:
        if ent.label_ not in tipos:
            continue
        nombre = _ARTICULO.sub("", _recortar_propn(ent))
        if not nombre:
            registro["eliminadas"]["sin PROPN"].append(ent.text)
        elif not _es_valida(nombre):
            registro["eliminadas"]["no válida"].append(nombre)
        else:
            c = candidatas.setdefault(_clave(nombre), {"n": 0, "etiquetas": Counter(), "formas": Counter()})
            c["n"] += 1
            c["etiquetas"][ent.label_] += 1
            c["formas"][nombre] += 1
    for c in candidatas.values():
        c["etiqueta"] = c["etiquetas"].most_common(1)[0][0]

    # 2. Glosario: prioridad. Sus menciones se cuentan en el texto y absorben
    # las entidades parecidas.
    salida = {t: [] for t in tipos}
    for termino in (glosario or []):
        n = len(re.findall(rf"(?<!\w){re.escape(termino)}(?!\w)", texto, re.IGNORECASE))
        clave_g = _clave(termino)
        for clave in [k for k in candidatas if fuzz.ratio(k, clave_g) >= umbral_fusion]:
            n += candidatas[clave]["n"] if clave != clave_g else 0
            if clave != clave_g:
                registro["fusiones"].append((candidatas[clave]["formas"].most_common(1)[0][0], candidatas[clave]["etiqueta"], termino, "ORG", "glosario"))
            del candidatas[clave]
        if n and "ORG" in salida:
            salida["ORG"].append((termino, n))

    # 3. Unir variantes parecidas
    canonicas, fusiones = _unificar(candidatas, umbral_fusion)
    registro["fusiones"] += fusiones
    for c in canonicas:
        if c["n"] >= minimo and c["etiqueta"] in salida:
            salida[c["etiqueta"]].append((c["formas"].most_common(1)[0][0], c["n"]))
    return {t: sorted(l, key=lambda e: (-e[1], e[0].lower())) for t, l in salida.items()}


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


# Absoluta: señala una fecha concreta (número/mes/año), o solo el día del
# mes ("el 23": absoluta parcial). El resto es relativa.
_ABSOLUTA = re.compile(rf"\d{{1,2}}[/-]\d{{1,2}}|\b(?:{_MESES})\b|\b(?:19|20)\d{{2}}\b|^el\s+{_NUM}$", re.IGNORECASE)
# Débil: unidad de tiempo sin referencia ("el mes", "este año", "la semana")
_DEBIL = re.compile(r"^(?:el|la|este|esta)\s+(?:semana|mes|a[ñn]o)$", re.IGNORECASE)
_PREFIJO = re.compile(r"^(?:(?:para|hasta|desde)\s+)?", re.IGNORECASE)
_DIA_SEMANA = re.compile(rf"^(?:el|este|esta)\s+((?:(?:pr[oó]ximo|pasado|siguiente)\s+)?(?:{_DIAS})(?:\s+{_NUM})?)$", re.IGNORECASE)


def _normalizar_fecha(expresion: str) -> str:
    """Forma única de una expresión: "hasta el 23", "para el 23" y "al 23" ->
    "el 23"; "el lunes" y "este lunes" -> "lunes"; "en 2026" -> "2026"."""
    e = re.sub(r"\s+", " ", expresion.strip().lower())
    e = _PREFIJO.sub("", e)
    e = re.sub(r"^(?:al|del)\s+", "el ", e)
    e = re.sub(r"^en\s+(?=(?:19|20)\d{2}$)", "", e)
    m = _DIA_SEMANA.match(e)
    return m.group(1) if m else e


def extraer_fechas(texto: str, registro=None) -> dict:
    """Expresiones de fecha: {"absolutas": [(expresión, frecuencia)], "relativas": [...]}.

    Absolutas: "20 de marzo", "12/10/2026", "2026" y el día del mes ("el 23").
    Relativas: "mañana", "lunes", "la próxima semana".
    Cada mención se cuenta una sola vez (no hay coincidencias solapadas) y las
    variantes se unen ("hasta el 23" = "el 23"; "el lunes" = "lunes").
    Se descartan las débiles ("el mes", "este año"); `registro` (dict opcional)
    recibe "eliminadas" con ellas.
    """
    conteo = {"absolutas": Counter(), "relativas": Counter()}
    eliminadas = Counter()
    for m in _FECHAS.finditer(texto):   # finditer no devuelve coincidencias solapadas
        expresion = _normalizar_fecha(m.group(0))
        if _DEBIL.match(expresion):
            eliminadas[expresion] += 1
            continue
        tipo = "absolutas" if _ABSOLUTA.search(expresion) else "relativas"
        conteo[tipo][expresion] += 1
    if registro is not None:
        registro["eliminadas"] = sorted(eliminadas.items(), key=lambda e: -e[1])
    return {t: sorted(c.items(), key=lambda e: (-e[1], e[0])) for t, c in conteo.items()}


def exportar_entidades_csv(entidades: dict, ruta) -> None:
    """CSV para revisar a mano: entidad, tipo, frecuencia y `correcta` (vacía;
    se llena con s/n). Se usa luego en evaluacion.evaluar_precision_entidades."""
    import csv
    from pathlib import Path
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel abre bien las tildes
        w = csv.writer(f)
        w.writerow(["entidad", "tipo", "frecuencia", "correcta"])
        for tipo, lista in entidades.items():
            for entidad, n in lista:
                w.writerow([entidad, tipo, n, ""])
