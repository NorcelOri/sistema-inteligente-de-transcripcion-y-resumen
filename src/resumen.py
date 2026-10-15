"""Resúmenes automáticos: mT5 (map-reduce) y TextRank (sumy).

El texto de entrada es el texto limpio (sin muletillas). mT5 XL-Sum acepta unos
512 tokens, así que se resume por bloques (map) y los resúmenes parciales se
unen y, si pasan del largo objetivo, se vuelven a resumir (reduce).
"""

import re
import time

from src.evaluacion import normalizar

MODELO = "csebuetnlp/mT5_multilingual_XLSum"
PALABRAS_MIN, PALABRAS_MAX = 60, 300     # límites absolutos del resumen
PROPORCION = (0.10, 0.15)                # el resumen mide ~10-15 % de la entrada


def objetivo_para(n_palabras: int) -> tuple:
    """Rango (mín, máx) de palabras del resumen según el largo de la entrada.

    10-15 % de la entrada, con mínimo 60 y máximo 300; el mínimo no pasa de 150.
    Con reuniones largas (>= 2000 palabras) da 150-300, como antes.
    """
    bajo = min(150, max(PALABRAS_MIN, round(PROPORCION[0] * n_palabras)))
    alto = min(PALABRAS_MAX, max(bajo + 30, round(PROPORCION[1] * n_palabras)))
    return bajo, alto
TOKENS_BLOQUE = 250          # tamaño de cada bloque de entrada (el modelo admite 512); 250 fue el único que dejó las 3 reuniones en 150-300 palabras con verificación
MAX_ENTRADA = 512
NUM_BEAMS = 4
NO_REPETIR_NGRAMA = 2
MAX_SALIDA = 84              # tokens de salida por bloque (valor de la ficha del modelo)
MIN_SALIDA = 0

# Tal como aparece en la ficha del modelo: colapsa saltos de línea y espacios
WHITESPACE_HANDLER = lambda k: re.sub(r"\s+", " ", re.sub(r"\n+", " ", k.strip()))

_modelo = {}


def _dtype_cuda():
    """bf16 si la GPU lo soporta (mT5 es inestable en fp16); si no, fp32."""
    import torch
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32


def cargar_mt5(dtype=None):
    """Carga tokenizer y modelo una sola vez (se vuelve a cargar si cambia el dtype)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dtype = dtype or _dtype_cuda()
    if _modelo.get("dtype") != dtype:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        _modelo["tokenizer"] = AutoTokenizer.from_pretrained(MODELO)
        _modelo["modelo"] = AutoModelForSeq2SeqLM.from_pretrained(MODELO, dtype=dtype).to(dispositivo).eval()
        _modelo["dtype"], _modelo["dispositivo"] = dtype, dispositivo
    return _modelo["tokenizer"], _modelo["modelo"], _modelo["dispositivo"]


def _resumir_bloque(texto: str, max_salida=MAX_SALIDA, min_salida=MIN_SALIDA, dtype=None,
                    prohibidas=()) -> str:
    """Resumen mT5 de un bloque (con el WHITESPACE_HANDLER de la ficha del modelo).
    `prohibidas`: palabras que el modelo no puede generar (bad_words_ids)."""
    import torch
    tokenizer, modelo, dispositivo = cargar_mt5(dtype)
    entrada = tokenizer(WHITESPACE_HANDLER(texto), return_tensors="pt",
                        truncation=True, max_length=MAX_ENTRADA).to(dispositivo)
    prohibidas_ids = [tokenizer(v, add_special_tokens=False).input_ids
                      for w in prohibidas for v in (w, " " + w)] or None
    with torch.no_grad():
        salida = modelo.generate(**entrada, max_length=max_salida, min_length=min_salida,
                                 num_beams=NUM_BEAMS, no_repeat_ngram_size=NO_REPETIR_NGRAMA,
                                 bad_words_ids=prohibidas_ids)
    resumen = tokenizer.decode(salida[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    if not resumen and _modelo["dtype"] != torch.float32:
        # Salida vacía = inestabilidad numérica (típica de mT5 en fp16): se repite en fp32
        return _resumir_bloque(texto, max_salida, min_salida, torch.float32, prohibidas)
    return resumen


def dividir_bloques(texto: str, tokens_bloque=TOKENS_BLOQUE) -> list:
    """Bloques de ~tokens_bloque tokens (del tokenizer de mT5) sin cortar oraciones."""
    from src.texto import segmentar_oraciones
    tokenizer = cargar_mt5()[0]
    bloques, actual, tokens = [], [], 0
    for oracion in segmentar_oraciones(texto):
        n = len(tokenizer.encode(oracion, add_special_tokens=False))
        if actual and tokens + n > tokens_bloque:
            bloques.append(" ".join(actual))
            actual, tokens = [], 0
        actual.append(oracion)
        tokens += n
    if actual:
        bloques.append(" ".join(actual))
    return bloques


def contar_palabras(texto: str) -> int:
    return len(texto.split())


def _mejor_oracion(bloque: str) -> str:
    """La oración mejor puntuada por TextRank dentro del bloque."""
    from sumy.nlp.stemmers import Stemmer
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words
    documento = PlaintextParser.from_string(bloque, Tokenizer("spanish")).document
    if len(documento.sentences) <= 1:
        return bloque.strip()
    resumidor = TextRankSummarizer(Stemmer("spanish"))
    resumidor.stop_words = get_stop_words("spanish")
    return str(resumidor(documento, 1)[0])


def _es_titular(parcial: str) -> bool:
    """Pregunta corta tipo titular ("¿Qué tienen en común X y Y?"): ruido típico de XL-Sum."""
    return parcial.rstrip().endswith("?") and contar_palabras(parcial) < 12


def resumir_mt5(texto: str, objetivo=None, tokens_bloque=TOKENS_BLOQUE,
                max_salida=MAX_SALIDA, min_salida=MIN_SALIDA, dtype=None, registro=None,
                descartar_preguntas=True, prohibidas=(), verificar=True) -> str:
    """Map-reduce con mT5 (con verificación de fidelidad por defecto; ver PLAN.md).

    verificar=True (por defecto): cada parcial se compara con su bloque; si trae nombres propios,
    números o fechas que el bloque no tiene, se reemplaza por la mejor oración de
    TextRank de ese bloque (un solo intento). `registro` recibe "bloques" y
    "sustituidos" (nivel 1).
 `registro` (dict opcional) recibe "parciales" (nivel 1)
    y "niveles" (lista con los resúmenes de cada nivel de reducción).

    Reduce: si la unión pasa del máximo, se resume otra vez; si eso la deja por
    debajo del mínimo (sobrecompresión) o no la achica, se conserva el nivel
    anterior y se poda quitando los parciales más cortos hasta entrar en el rango.
    """
    registro = registro if registro is not None else {}
    objetivo = objetivo or objetivo_para(contar_palabras(texto))
    def resumir_b(b, nivel1=False):
        p = _resumir_bloque(b, max_salida, min_salida, dtype, prohibidas)
        if verificar and (not p or any(verificar_alucinaciones(p, b).values())):
            p = _mejor_oracion(b)
            if nivel1:
                registro["sustituidos"] += 1
        return p

    bloques = dividir_bloques(texto, tokens_bloque)
    registro["bloques"], registro["sustituidos"] = len(bloques), 0
    parciales = [resumir_b(b, True) for b in bloques]
    registro["parciales"] = parciales
    registro["niveles"] = []
    actual = [p for p in parciales if p and not (descartar_preguntas and _es_titular(p))]
    union = lambda lista: " ".join(lista)
    while contar_palabras(union(actual)) > objetivo[1] and len(registro["niveles"]) < 3:
        nuevos = [resumir_b(b) for b in dividir_bloques(union(actual), tokens_bloque)]
        registro["niveles"].append(nuevos)
        if not (objetivo[0] <= contar_palabras(union(nuevos)) < contar_palabras(union(actual))):
            break
        actual = nuevos
    # Poda: quita los parciales más cortos (menos información) conservando el orden
    while contar_palabras(union(actual)) > objetivo[1] and len(actual) > 1:
        actual.remove(min(actual, key=contar_palabras))
    return union(actual)


def oraciones_para_objetivo(texto: str, objetivo) -> int:
    """Cuántas oraciones tomar para quedar en el rango objetivo, según el largo
    medio de las oraciones del texto (apunta a la mitad del rango)."""
    from src.texto import segmentar_oraciones
    oraciones = segmentar_oraciones(texto)
    medio = max(1, sum(contar_palabras(o) for o in oraciones) / max(1, len(oraciones)))
    return max(1, min(len(oraciones), round(sum(objetivo) / 2 / medio)))


def resumir_textrank(texto: str, n_oraciones: int = None, objetivo=None) -> str:
    """TextRank (sumy, idioma spanish). Si no se da n_oraciones, se calcula para
    llegar a 150-300 palabras; luego se ajusta si quedó fuera del rango."""
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
    resumidor = TextRankSummarizer(Stemmer("spanish"))
    resumidor.stop_words = get_stop_words("spanish")

    def extraer(n):
        elegidas = {str(o) for o in resumidor(parser.document, n)}
        # sumy no garantiza el orden: se devuelve en el orden original del texto
        return " ".join(str(o) for o in parser.document.sentences if str(o) in elegidas)

    if n_oraciones is not None:
        return extraer(n_oraciones)
    objetivo = objetivo or objetivo_para(contar_palabras(texto))
    n = oraciones_para_objetivo(texto, objetivo)
    resumen = extraer(n)
    # Ajuste fino: sube o baja de a una oración hasta entrar en el rango
    while contar_palabras(resumen) > objetivo[1] and n > 1:
        n -= 1
        resumen = extraer(n)
    while contar_palabras(resumen) < objetivo[0] and n < len(parser.document.sentences):
        n += 1
        candidato = extraer(n)
        if contar_palabras(candidato) > objetivo[1]:
            break
        resumen = candidato
    return resumen


def resumir(texto: str, metodo: str = "textrank", **kwargs) -> str:
    """Resume el texto con "textrank" (por defecto) o "mt5" (map-reduce; inventa datos, ver PLAN.md)."""
    if metodo == "mt5":
        return resumir_mt5(texto, **kwargs)
    if metodo == "textrank":
        return resumir_textrank(texto, **kwargs)
    raise ValueError(f'Método desconocido: {metodo!r} (usa "mt5" o "textrank")')


# ---------------------------------------------------------------------------
# Verificación de alucinaciones
# ---------------------------------------------------------------------------

_NUMERO = re.compile(r"\d+(?:[.,:]\d+)*")


def _palabras_norm(texto: str) -> set:
    return set(normalizar(texto, quitar_tildes=True).split())


def verificar_alucinaciones(resumen: str, transcripcion: str) -> dict:
    """Nombres propios, números y fechas del resumen que NO están en la transcripción.

    - Nombres: tokens PROPN de spaCy del resumen (sin tildes ni mayúsculas) que
      no aparecen como palabra en la transcripción.
    - Números: cifras del resumen ausentes en la transcripción.
    - Fechas: expresiones de fecha del resumen (extraer_fechas) cuya forma
      normalizada no aparece en la transcripción.
    Es una señal para revisar, no un veredicto: un número escrito con letras en la
    transcripción ("veinte") o una fecha parafraseada se marcarán como ausentes.
    """
    from src.nlp import extraer_fechas
    from src.texto import _cargar_spacy

    vocab = _palabras_norm(transcripcion)
    texto_norm = normalizar(transcripcion, quitar_tildes=True)

    nombres = []
    for tok in _cargar_spacy()(resumen):
        if tok.pos_ == "PROPN" and normalizar(tok.text, True) not in vocab and tok.text not in nombres:
            nombres.append(tok.text)

    numeros_trans = set(_NUMERO.findall(transcripcion))
    numeros = [n for n in dict.fromkeys(_NUMERO.findall(resumen)) if n not in numeros_trans]

    fechas = []
    for tipo, lista in extraer_fechas(resumen).items():
        for expresion, _ in lista:
            if normalizar(expresion, True) not in texto_norm:
                fechas.append(expresion)
    return {"nombres": nombres, "numeros": numeros, "fechas": fechas}


def medir(texto: str, metodo: str, **kwargs) -> dict:
    """Resume y mide: palabras, % de reducción, tiempo (s). Para la evaluación."""
    inicio = time.perf_counter()
    resumen = resumir(texto, metodo, **kwargs)
    segundos = time.perf_counter() - inicio
    p_texto, p_resumen = contar_palabras(texto), contar_palabras(resumen)
    return {"resumen": resumen, "palabras": p_resumen, "segundos": round(segundos, 1),
            "reduccion": round(100 * (1 - p_resumen / p_texto), 1)}
