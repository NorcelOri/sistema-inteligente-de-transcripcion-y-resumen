"""Lectura opcional de glosario y variantes desde archivos.

El sistema es general: NO hay glosario ni variantes por defecto. El usuario
puede pasar una lista de términos (o un archivo con uno por línea). Hay un
ejemplo en config/ejemplos/caso_reuniones/.
"""

from pathlib import Path


def _lineas(ruta):
    """Líneas útiles de un archivo: sin vacías ni comentarios (#)."""
    lineas = Path(ruta).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lineas if l.strip() and not l.strip().startswith("#")]


def cargar_glosario(ruta=None) -> list:
    """Lista de términos (nombres propios, siglas) de un archivo. Sin ruta: []."""
    return _lineas(ruta) if ruta else []


def cargar_variantes(ruta=None) -> dict:
    """Diccionario {variante en minúsculas: forma correcta}. Sin ruta: {}.

    Formato de cada línea:  variante => correcta
    """
    variantes = {}
    for linea in (_lineas(ruta) if ruta else []):
        if "=>" in linea:
            mala, buena = (p.strip() for p in linea.split("=>", 1))
            variantes[mala.lower()] = buena
    return variantes
