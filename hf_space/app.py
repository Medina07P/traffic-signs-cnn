import json
import os

import gradio as gr
import numpy as np
from PIL import Image

from utils import load_cnn, preprocess_image

KB_PATH = os.path.join(os.path.dirname(__file__), "traffic_signs_es.json")
QA_MODEL = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
RETRIEVER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONFIDENCE_THRESHOLD = 0.15

print("Cargando modelo CNN desde Hugging Face Hub...")
cnn = load_cnn()

print("Cargando base de conocimiento...")
with open(KB_PATH, encoding="utf-8") as f:
    KB: dict = json.load(f)

LABEL_MAP = {int(k): v["nombre"] for k, v in KB.items()}

print("Cargando modelos Transformer...")
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

qa_pipe = pipeline("question-answering", model=QA_MODEL)
retriever = SentenceTransformer(RETRIEVER_MODEL)

KB_KEYS = list(KB.keys())
KB_TEXTS = [f"{v['nombre']}. {v['descripcion']}. {v['significado']}." for v in KB.values()]
KB_EMBEDDINGS = retriever.encode(KB_TEXTS, convert_to_tensor=True)
print("Todos los modelos cargados correctamente.")

last_detected = {"class_id": None}


def classify(image: Image.Image):
    if image is None:
        return "Por favor, sube o captura una imagen."
    arr = preprocess_image(image)
    pred = cnn.predict(arr, verbose=0)[0]
    cid = int(np.argmax(pred))
    conf = float(pred[cid])
    last_detected["class_id"] = cid
    nombre = LABEL_MAP.get(cid, f"Clase {cid}")
    return f"{nombre}\nConfianza: {conf:.1%}"


def ask_agent(history, question: str):
    if not question.strip():
        return history, ""

    cid = last_detected["class_id"]

    if cid is None:
        q_emb = retriever.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, KB_EMBEDDINGS)[0]
        best_idx = int(scores.argmax())
        class_key = KB_KEYS[best_idx]
    else:
        class_key = str(cid)

    entry = KB[class_key]
    context = (
        f"{entry['nombre']}. {entry['descripcion']}. "
        f"{entry['significado']}. {entry['accion']}. {entry['contexto']}"
    )

    result = qa_pipe(question=question, context=context)
    if result["score"] < CONFIDENCE_THRESHOLD:
        answer = "No tengo información suficiente sobre esa consulta."
    else:
        answer = f"{result['answer']} (Fuente: {entry['nombre']})"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


with gr.Blocks(title="Detector de Señales de Tránsito 🚦", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚦 Detector de Señales de Tránsito")
    gr.Markdown(
        "Detecta y clasifica señales del dataset GTSRB con una CNN entrenada en TensorFlow/Keras. "
        "Luego pregunta al agente Q&A basado en BERT sobre cualquier señal."
    )

    with gr.Tab("Detectar señal"):
        with gr.Row():
            img_input = gr.Image(
                sources=["webcam", "upload"],
                type="pil",
                label="Captura o sube una imagen de señal de tránsito",
            )
            result_box = gr.Textbox(label="Resultado de la clasificación", lines=3)
        detect_btn = gr.Button("Clasificar", variant="primary")
        detect_btn.click(classify, inputs=img_input, outputs=result_box)
        img_input.change(classify, inputs=img_input, outputs=result_box)

    with gr.Tab("Preguntar al agente"):
        gr.Markdown(
            "Haz preguntas sobre la señal detectada (o cualquier señal de tránsito). "
            "El agente usa BERT en español para responder."
        )
        chatbot = gr.Chatbot(label="Agente Q&A de Señales", height=400, type="messages")
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ejemplo: ¿Qué significa esta señal? ¿Qué debo hacer?",
                label="Tu pregunta",
                scale=4,
            )
            send_btn = gr.Button("Enviar", variant="primary", scale=1)
        clear_btn = gr.Button("Limpiar conversación")

        send_btn.click(ask_agent, inputs=[chatbot, msg_box], outputs=[chatbot, msg_box])
        msg_box.submit(ask_agent, inputs=[chatbot, msg_box], outputs=[chatbot, msg_box])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

    gr.Markdown(
        "**Proyecto Final IA — FUP | Tecnología:** CNN (TensorFlow/Keras) + "
        "BERT Q&A (Hugging Face Transformers) + Sentence-Transformers"
    )

if __name__ == "__main__":
    demo.launch()
