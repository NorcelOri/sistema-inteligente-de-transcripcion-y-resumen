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


# ---------------------------------------------------------------------------
# Limpieza para análisis
#
# Whisper casi nunca transcribe interjecciones puras (eh, em, mmm), pero sí deja
# marcadores de relleno ("o sea", "bueno", "digamos", "este"). Solo se borran
# cuando el contexto lo deja claro: aislados entre pausas (comas, puntos, inicio
# de texto) o repetidos. "bueno" en "me parece bueno", "este" en "este evento"
# o "pues" en medio de una frase no se tocan.
# ---------------------------------------------------------------------------

# Interjecciones: se borran siempre (palabra completa).
_INTERJECCION = re.compile(r"(?<!\w)(?:e+h+m*|e+m+|h?m{2,}|hm+|u+h+m+|u+m+)(?!\w)[,.]?\s*", re.IGNORECASE)

# Marcadores: solo si van aislados por pausas. Los que cierran con coma ("bueno,")
# o también con punto ("digamos.").
_MARCADORES_COMA = r"o\s+sea|bueno|pues|este|digamos"
_MARCADORES_PUNTO = r"digamos|o\s+sea"
_MARCADOR = re.compile(
    r"(?P<pre>^|[.!?…]\s+|,\s+|(?<=\w)\s+(?=(?:y|pero)\s))"  # inicio, fin de oración, coma, o tras 'y'/'pero'
    r"(?P<conj>(?:y|pero)\s+)?"
    r"(?P<m>(?:" + _MARCADORES_COMA + r")\s*,|(?:" + _MARCADORES_PUNTO + r")\s*\.)\s*",
    re.IGNORECASE,
)

# Muletilla de cierre: "..., ¿no?" -> "..."
_COLA_NO = re.compile(r",\s*¿no\?", re.IGNORECASE)

# Palabras funcionales que, repetidas seguidas, son tartamudeo ("que que", "el el").
# No incluye "sí", "no", "ya", "muy"... que se repiten con sentido.
_FUNCIONALES = (r"que|el|la|los|las|un|una|de|del|en|y|a|se|lo|es|este|esta|con|por|para|"
                r"mi|su|me|te|al|como|si|más")
_REPETIDA = re.compile(r"(?<!\w)(" + _FUNCIONALES + r")(?:[\s,]+\1)+(?!\w)", re.IGNORECASE)


def _mayuscula_inicial(m: re.Match) -> str:
    return m.group(0)[:-1] + m.group(0)[-1].upper()


def limpiar_texto_con_registro(texto: str):
    """Limpia el texto y devuelve (texto_limpio, borrados).

    borrados: lista de {"tipo", "borrado", "contexto"} para revisar falsos positivos.
    """
    borrados = []

    def anotar(tipo, m, texto_completo):
        borrados.append({"tipo": tipo, "borrado": m.group(0).strip(" ,.")
                         if tipo != "repetida" else m.group(0),
                         "contexto": texto_completo[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")})

    # 1. Interjecciones
    original = texto
    for m in _INTERJECCION.finditer(original):
        anotar("interjección", m, original)
    texto = _INTERJECCION.sub("", texto)

    # 2. Marcadores aislados. Se repite hasta que no cambie ("Bueno, o sea, ...").
    cambio = True
    while cambio:
        cambio = False
        m = _MARCADOR.search(texto)
        if m:
            anotar("marcador", m, texto)
            pre = m.group("pre").strip() if m.group("pre") else ""
            conj = m.group("conj") or ""
            resto = texto[m.end():]
            # Tras fin de oración o inicio, la palabra siguiente pasa a mayúscula
            if (pre in {"", ".", "!", "?", "…"}) and not conj and resto:
                resto = resto[0].upper() + resto[1:]
            separador = (m.group("pre") or "") if not (m.group("pre") or "").isspace() else " "
            # "..., digamos. Siguiente" cierra la oración: la coma pasa a punto
            if pre == "," and not conj and m.group("m").rstrip().endswith(".") and resto:
                separador = ". "
                resto = resto[0].upper() + resto[1:]
            texto = texto[:m.start()] + separador + conj + resto
            cambio = True

    # 3. "..., ¿no?"
    for m in _COLA_NO.finditer(texto):
        anotar("cola ¿no?", m, texto)
    texto = _COLA_NO.sub(".", texto)

    # 4. Palabras funcionales repetidas
    for m in _REPETIDA.finditer(texto):
        anotar("repetida", m, texto)
    texto = _REPETIDA.sub(lambda m: m.group(1), texto)

    return _normalizar_signos(texto), borrados


def _normalizar_signos(texto: str) -> str:
    """Espacios y signos: sin dobles espacios ni espacio antes de la puntuación,
    un espacio después, sin signos duplicados. Conserva la puntuación original."""
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s+([,.;:!?…])", r"\1", texto)           # sin espacio antes
    texto = re.sub(r",(\s*,)+", ",", texto)                   # ", ," -> ","
    texto = re.sub(r"([.!?])(\s*\.)+(?!\.)", r"\1", texto)    # ". ." -> "."
    texto = re.sub(r"(?<=\d)\s*:\s*(?=\d)", ":", texto)       # horas 6 : 20 -> 6:20
    texto = re.sub(r"([,;])(?=[^\s\d])", r"\1 ", texto)       # espacio tras coma
    texto = re.sub(r"(?<![A-ZÁÉÍÓÚÑ])([.!?])(?=[A-ZÁÉÍÓÚÑ¿¡])", r"\1 ", texto)
    texto = re.sub(r"¿\s+", "¿", texto)
    texto = re.sub(r"¡\s+", "¡", texto)
    texto = re.sub(r"^[\s,;.]+", "", texto)                   # sin signos al inicio
    return texto.strip()


def limpiar_texto(texto: str) -> str:
    """Texto limpio para análisis: sin muletillas claras y con signos normalizados."""
    return limpiar_texto_con_registro(texto)[0]


def tokenizar(texto: str) -> list:
    """Palabras del texto (minúsculas, sin puntuación)."""
    return [p.lower() for p in _PALABRA.findall(texto)]


def _cargar_spacy():
    """es_core_news_md completo (parser para oraciones, NER para entidades)."""
    if "oraciones" not in _spacy:
        import spacy
        _spacy["oraciones"] = spacy.load("es_core_news_md")
        _spacy["oraciones"].max_length = 2_000_000
    return _spacy["oraciones"]


def segmentar_oraciones(texto: str) -> list:
    """Divide el texto en oraciones con spaCy (es_core_news_md)."""
    doc = _cargar_spacy()(texto)
    return [o.text.strip() for o in doc.sents if o.text.strip()]


def preparar_textos(resultado, glosario=None, variantes=None) -> dict:
    """De una transcripción (con reparación) a los textos del sistema.

    - "fiel": lo que se entrega como transcripción (con nombres corregidos).
    - "limpio": para análisis (sin muletillas); nunca se entrega como transcripción.
    - "oraciones": el texto limpio dividido en oraciones.
    - "borrados": lo que se quitó, para revisión.
    """
    fiel = corregir_nombres(resultado["texto"], glosario, variantes)
    limpio, borrados = limpiar_texto_con_registro(fiel)
    return {"fiel": fiel, "limpio": limpio,
            "oraciones": segmentar_oraciones(limpio), "borrados": borrados}
