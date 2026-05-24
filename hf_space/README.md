---
title: Detector de Señales de Tránsito
emoji: 🚦
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "4.16.0"
app_file: app.py
pinned: false
license: mit
---

# Detector de Señales de Tránsito

Aplicación de visión computacional que detecta y clasifica señales de tránsito del dataset GTSRB (43 clases) usando una CNN entrenada con TensorFlow/Keras, y permite hacer preguntas sobre las señales usando un Transformer Q&A en español.

## Cómo usar

1. **Pestaña "Detectar señal":** Sube una foto o captura con la cámara. La CNN clasificará la señal y mostrará su nombre en español y el porcentaje de confianza.
2. **Pestaña "Preguntar al agente":** Escribe una pregunta sobre la señal detectada (o cualquier señal de tránsito). El agente basado en BERT responderá usando su base de conocimiento.

## Tecnologías

- CNN: TensorFlow/Keras, entrenada en GTSRB (43 clases)
- Agente Q&A: `mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es`
- Retrieval semántico: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Interfaz: Gradio
