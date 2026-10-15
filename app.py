"""Interfaz Gradio. Lanzar: python app.py  (http://127.0.0.1:7860)"""

import gradio as gr

from src.pipeline import procesar

METODOS = {"TextRank (recomendado)": "textrank", "mT5 verificado (experimental)": "mt5"}


def _lista(pares):
    return "\n".join(f"- {e} ({n})" for e, n in pares) or "—"


def ejecutar(archivo, glosario, metodo, ruido, progress=gr.Progress()):
    if archivo is None:
        raise gr.Error("Sube un archivo de audio o video (.mp3, .wav, .m4a, .mp4).")
    terminos = (glosario or "").split(",")
    try:
        r = procesar(archivo, glosario=terminos, metodo=METODOS[metodo], reducir_ruido=ruido,
                     progreso=lambda f, d: progress(f, desc=d))
    except ValueError as e:
        raise gr.Error(str(e))
    ent = r["entidades"]
    info = (f"### Personas\n{_lista(ent.get('PER', []))}\n\n### Lugares\n{_lista(ent.get('LOC', []))}\n\n"
            f"### Organizaciones\n{_lista(ent.get('ORG', []))}\n\n"
            f"### Fechas absolutas\n{_lista(r['fechas']['absolutas'])}\n\n"
            f"### Fechas relativas\n{_lista(r['fechas']['relativas'])}")
    claves = ("### Palabras clave\n" + ", ".join(p for p, _ in r["palabras_clave"]) +
              "\n\n### Temas\n" + "\n".join(f"- {t[0]}" for t in r["temas"]))
    m = r["metadatos"]
    meta = (f"- Duración: {m['duracion']}\n- Palabras: {m['palabras']}\n- Oraciones: {m['oraciones']}\n"
            f"- Método de resumen: {r['metodo']}\n\n### Tiempos por etapa (s)\n" +
            "\n".join(f"- {k}: {v}" for k, v in r["tiempos"].items()))
    return r["fiel"], r["resumen"], claves, info, meta, [str(r["ruta_txt"]), str(r["ruta_pdf"])]


with gr.Blocks(title="Transcripción y análisis de reuniones") as demo:
    gr.Markdown("# Transcripción y análisis de audio/video")
    with gr.Row():
        with gr.Column():
            archivo = gr.File(label="Audio o video", file_types=[".mp3", ".wav", ".m4a", ".mp4"], type="filepath")
            glosario = gr.Textbox(label="Glosario (opcional, términos separados por comas)")
            metodo = gr.Radio(list(METODOS), value="TextRank (recomendado)", label="Método de resumen")
            ruido = gr.Checkbox(label="Reducir ruido")
            boton = gr.Button("Procesar", variant="primary")
        with gr.Column():
            descargas = gr.Files(label="Descargas (.txt y .pdf)")
    with gr.Tabs():
        with gr.Tab("Transcripción"):
            t_trans = gr.Textbox(lines=18, label="Transcripción")
        with gr.Tab("Resumen"):
            t_res = gr.Textbox(lines=10, label="Resumen")
        with gr.Tab("Palabras clave y temas"):
            t_claves = gr.Markdown()
        with gr.Tab("Información relevante"):
            t_info = gr.Markdown()
        with gr.Tab("Metadatos y tiempos"):
            t_meta = gr.Markdown()
    boton.click(ejecutar, [archivo, glosario, metodo, ruido],
                [t_trans, t_res, t_claves, t_info, t_meta, descargas])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
