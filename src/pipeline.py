"""Flujo completo: audio/video -> transcripción -> análisis -> resumen -> .txt y .pdf."""

import time
from pathlib import Path

from src import audio, nlp, resumen as R, salidas, texto, transcripcion

MODELO_WHISPER = "medium"
ETAPAS = ["Extrayendo audio", "Transcribiendo", "Corrigiendo y limpiando", "Palabras clave y temas",
          "Entidades y fechas", "Resumiendo", "Generando archivos"]


def _imprimir(fraccion, descripcion):
    print(f"[{fraccion:4.0%}] {descripcion}", flush=True)


def procesar(ruta, idioma="es", glosario=None, metodo="textrank", reducir_ruido=False,
             progreso=_imprimir, inicio=None, duracion=None) -> dict:
    """Procesa un audio o video y devuelve todos los resultados.

    glosario: lista de términos (o None). metodo: "textrank" o "mt5" (experimental).
    progreso(fraccion, descripcion) se llama al empezar cada etapa.
    inicio/duracion (s): procesar solo un fragmento. Si el archivo ya fue
    transcrito (data/transcripciones), la transcripción se reutiliza.
    Devuelve un dict con los resultados y "tiempos" (segundos por etapa).
    """
    ruta = Path(ruta)
    glosario = [g.strip() for g in (glosario or []) if g.strip()] or None
    tiempos, r = {}, {}
    marca = {}

    def etapa(i, clave):
        if marca:
            tiempos[marca["clave"]] = round(time.perf_counter() - marca["t"], 1)
        marca.update(t=time.perf_counter(), clave=clave)
        progreso(i / len(ETAPAS), ETAPAS[i] + "...")

    etapa(0, "audio")
    wav = audio.extraer_audio(ruta, inicio=inicio, duracion=duracion)
    if reducir_ruido:
        wav = audio.reducir_ruido(wav)
    r["duracion_s"] = audio.duracion_segundos(wav)

    etapa(1, "transcripcion")
    transcrito = transcripcion.transcribir(wav, idioma, MODELO_WHISPER, glosario=glosario)
    r["reparaciones"] = transcrito.get("reparaciones", [])

    etapa(2, "correccion_limpieza")
    textos = texto.preparar_textos(transcrito, glosario)
    r["fiel"], r["limpio"] = textos["fiel"], textos["limpio"]

    etapa(3, "palabras_clave_temas")
    r["palabras_clave"] = nlp.extraer_palabras_clave(r["limpio"], glosario=glosario)
    r["temas"] = nlp.extraer_temas(r["limpio"], glosario=glosario)

    etapa(4, "entidades_fechas")
    r["entidades"] = nlp.extraer_entidades(r["limpio"], glosario=glosario)
    r["fechas"] = nlp.extraer_fechas(r["limpio"])

    etapa(5, "resumen")
    r["resumen"] = R.resumir(r["limpio"], metodo)
    r["metodo"] = metodo

    etapa(6, "salidas")
    nombre = wav.stem
    r["metadatos"] = salidas.calcular_metadatos(r["fiel"], textos["oraciones"], r["duracion_s"])
    r["ruta_txt"] = salidas.guardar_txt(r["fiel"], nombre)
    r["ruta_pdf"] = salidas.generar_pdf(nombre, r["resumen"], r["palabras_clave"], r["temas"],
                                        r["entidades"], r["fechas"], r["metadatos"])
    tiempos[marca["clave"]] = round(time.perf_counter() - marca["t"], 1)

    tiempos["total"] = round(sum(tiempos.values()), 1)
    r["tiempos"] = tiempos
    progreso(1.0, "Listo")
    return r
