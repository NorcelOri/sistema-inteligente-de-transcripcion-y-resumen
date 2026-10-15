"""Salidas del sistema: metadatos, transcripción en .txt y reporte en PDF (fpdf2)."""

from pathlib import Path

from fpdf import FPDF

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = RAIZ / "resultados"

# Fuente Unicode (Arial de Windows); las fuentes base de fpdf2 no cubren bien tildes y ñ
FUENTE = Path(r"C:\Windows\Fonts\arial.ttf")
FUENTE_NEGRITA = Path(r"C:\Windows\Fonts\arialbd.ttf")


def calcular_metadatos(texto_fiel: str, oraciones: list, duracion_s: float) -> dict:
    """Duración (s y mm:ss), palabras de la transcripción y oraciones del texto limpio."""
    minutos, segundos = divmod(int(round(duracion_s)), 60)
    return {"duracion_s": round(duracion_s, 1), "duracion": f"{minutos}:{segundos:02d}",
            "palabras": len(texto_fiel.split()), "oraciones": len(oraciones)}


def guardar_txt(texto_fiel: str, nombre: str, carpeta=CARPETA_RESULTADOS) -> Path:
    """Guarda la transcripción fiel (UTF-8) y devuelve la ruta."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}_transcripcion.txt"
    ruta.write_text(texto_fiel, encoding="utf-8")
    return ruta


def _lista_entidades(entidades: dict) -> list:
    nombres = {"PER": "Personas", "LOC": "Lugares", "ORG": "Organizaciones"}
    return [(nombres.get(t, t), ", ".join(f"{e} ({n})" for e, n in lista) or "—")
            for t, lista in entidades.items()]


def generar_pdf(nombre: str, resumen: str, palabras_clave: list, temas: list,
                entidades: dict, fechas: dict, metadatos: dict, carpeta=CARPETA_RESULTADOS) -> Path:
    """PDF con resumen, palabras clave, temas, entidades, fechas y metadatos."""
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font("Arial", "", str(FUENTE))
    pdf.add_font("Arial", "B", str(FUENTE_NEGRITA))
    pdf.add_page()

    def titulo(texto, tam=13):
        pdf.set_font("Arial", "B", tam)
        pdf.ln(3)
        pdf.multi_cell(0, 8, texto, new_x="LMARGIN", new_y="NEXT")

    def parrafo(texto):
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, texto, new_x="LMARGIN", new_y="NEXT")

    titulo(f"Reporte: {nombre}", 16)
    titulo("Resumen")
    parrafo(resumen)
    titulo("Palabras clave")
    parrafo(", ".join(p for p, _ in palabras_clave) or "—")
    titulo("Temas principales")
    parrafo(", ".join(t[0] for t in temas) or "—")
    titulo("Información relevante")
    for etiqueta, valor in _lista_entidades(entidades):
        parrafo(f"{etiqueta}: {valor}")
    for etiqueta, clave in (("Fechas absolutas", "absolutas"), ("Fechas relativas", "relativas")):
        parrafo(f"{etiqueta}: " + (", ".join(f"{e} ({n})" for e, n in fechas.get(clave, [])) or "—"))
    titulo("Metadatos")
    parrafo(f"Duración: {metadatos['duracion']} ({metadatos['duracion_s']} s)\n"
            f"Palabras de la transcripción: {metadatos['palabras']}\n"
            f"Oraciones: {metadatos['oraciones']}")

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}_reporte.pdf"
    pdf.output(str(ruta))
    return ruta
