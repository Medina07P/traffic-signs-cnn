# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Proyecto final del curso de IA (Tercer Corte — FUP). **Tema 3: Detección de Señales de Tránsito para Conducción Autónoma** con asistente conversacional. Los requisitos completos están en `CNN FINAL.pdf`.

El sistema combina:
1. **CNN clasificadora** — 43 clases GTSRB, TensorFlow/Keras, entrenada en Kaggle (GPU T4).
2. **Agente Q&A Transformer** — BERT extractivo en español sobre knowledge base JSON de las 43 señales.
3. **App web Gradio** — desplegada en Hugging Face Spaces, accesible desde móvil vía navegador.

**App desplegada:** `https://huggingface.co/spaces/jarolmedina41/traffic-signs-app`  
**Modelo en HF Hub:** `https://huggingface.co/jarolmedina41/traffic-signs-cnn`

## Architecture

```
[Cámara móvil] ──HTTPS──► [HF Space Gradio]
                              │
                              ├─ Pestaña "Detectar señal":
                              │     gr.Image(webcam/upload) → CNN (.h5 cargado desde HF Hub)
                              │     → nombre clase ES + confianza
                              │
                              └─ Pestaña "Preguntar al agente":
                                    gr.Chatbot → BERT Q&A + retrieval semántico
                                    → respuesta basada en la señal detectada

[Kaggle Notebook] ──entrena──► cnn_gtsrb_v1.h5 ──upload──► HF Hub
                                                              ▲
                                                              └──── HF Space lo descarga al iniciar
```

```
PROYECTO/
├── src/
│   ├── config.py          # Fuente única de verdad: IMG_SIZE, LABEL_MAP, modelos HF, rutas
│   ├── model.py           # build_cnn() — arquitectura Keras Sequential
│   ├── data_loader.py     # load_gtsrb(), compute_class_weights()
│   ├── qa_agent.py        # TrafficSignAgent — Q&A extractivo + retrieval semántico
│   └── utils.py           # preprocess_image(), load_model_from_hub(), load_model_local()
├── notebooks/
│   ├── 01_train_cnn_kaggle.ipynb   # Autocontenido para Kaggle — entrenamiento end-to-end
│   └── 02_eda_gtsrb.ipynb          # Análisis exploratorio local
├── hf_space/              # App Gradio → subir completo a HF Spaces
│   ├── app.py             # Entry point Gradio
│   ├── utils.py           # preprocess_image() + load_cnn() independiente de src/
│   ├── traffic_signs_es.json  # Copia de la KB (necesaria en el Space)
│   └── requirements.txt   # Dependencias del Space (tensorflow-cpu)
├── data/knowledge_base/
│   └── traffic_signs_es.json  # 43 entradas: nombre, descripcion, significado, accion, contexto
└── models/                    # cnn_gtsrb_v1.h5 descargado (gitignored)
```

### Componente 1 — CNN Clasificadora

- **Archivos clave:** `src/model.py`, `src/data_loader.py`, `src/config.py`
- Entrada fija **32×32×3** (estándar GTSRB), normalización /255.
- Salida: softmax de **43 clases**.
- Arquitectura: 3 bloques Conv→Conv→BN→ReLU→MaxPool→Dropout, seguidos de Dense(256)+Dropout+softmax(43).
- GTSRB está desbalanceado; `data_loader.compute_class_weights()` calcula los pesos para `model.fit`.
- Entrenamiento en `notebooks/01_train_cnn_kaggle.ipynb` (Kaggle GPU T4). Modelo guardado como `cnn_gtsrb_v1.h5`.

### Componente 2 — Agente Q&A Transformer

- **Archivos clave:** `src/qa_agent.py`, `data/knowledge_base/traffic_signs_es.json`
- `TrafficSignAgent.__init__`: carga pipeline `question-answering` con `mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es` y `SentenceTransformer` para retrieval.
- `TrafficSignAgent.answer(question, current_class_id)`:
  - Si hay señal detectada → usa su entrada JSON como contexto.
  - Si no → búsqueda semántica con `sentence-transformers` sobre la KB completa.
  - Score < 0.15 → "No tengo información suficiente."

### Componente 3 — App Gradio (`hf_space/app.py`)

- Carga el CNN desde HF Hub al iniciar el Space (`hf_hub_download`).
- Pestaña 1: `gr.Image(sources=["webcam","upload"])` → `classify()` → nombre + confianza.
- Pestaña 2: `gr.Chatbot` + `gr.Textbox` → `ask_agent()` → respuesta Q&A.
- `last_detected` dict comparte la última clase entre las dos pestañas (state de proceso).
- `hf_space/` es **independiente de `src/`** — tiene su propio `utils.py` para poder desplegarse sin el paquete local.

## Commands

**Instalar dependencias locales:**
```bash
pip install -r requirements-local.txt
```

**Ejecutar app localmente antes de desplegar:**
```bash
cd hf_space
python app.py
# Abre http://localhost:7860
```

**Usar módulos src/ desde Python:**
```python
from src.config import LABEL_MAP
from src.model import build_cnn
from src.qa_agent import TrafficSignAgent
```

**Subir modelo a HF Hub (después de entrenar en Kaggle):**
```bash
huggingface-cli login
huggingface-cli repo create traffic-signs-cnn --type model
huggingface-cli upload jarolmedina41/traffic-signs-cnn models/cnn_gtsrb_v1.h5
```

**Desplegar en HF Spaces:**
```bash
git clone https://huggingface.co/spaces/jarolmedina41/traffic-signs-app
# Copiar contenido de hf_space/ al repo clonado
git add . && git commit -m "Deploy Gradio app" && git push
```

**Explorar notebooks:**
```bash
jupyter notebook notebooks/
```

## Conventions

- `src/config.py` es la única fuente de verdad: `IMG_SIZE=32`, `NUM_CLASSES=43`, `LABEL_MAP`, nombres de modelos HF, paths. Nunca hardcodear estos valores en otros archivos.
- `hf_space/` es autocontenido — no importa desde `src/`. Duplica `utils.py` mínimo intencionalmente.
- `data/knowledge_base/traffic_signs_es.json` y `hf_space/traffic_signs_es.json` son copias sincronizadas.
- Todos los textos expuestos al usuario en **español**.
- `models/` está en `.gitignore`; modelo versionado por nombre (`cnn_gtsrb_v1.h5`, nunca sobreescribir).
- `requirements-local.txt` para desarrollo; `requirements-train.txt` para Kaggle; `hf_space/requirements.txt` usa `tensorflow-cpu` (no GPU en HF Spaces gratis).

## Dataset Notes

GTSRB: 43 clases, ~50.000 imágenes de entrenamiento, distribución desbalanceada (200–2000 imágenes por clase). `data_loader.compute_class_weights()` resuelve el desbalance. El split test viene en `Test.csv` (separador `;`, columnas `Path` y `ClassId`). Train tiene subcarpetas `0/` a `42/` con imágenes `.png`.

Descarga en Kaggle: dataset `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign` disponible directamente en notebooks Kaggle sin descargar manualmente.

## Transformer Q&A Notes

- Modelo principal: `mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es` (~250 MB CPU).
- Alternativa más potente si los recursos lo permiten: `PlanTL-GOB-ES/roberta-base-bne-sqac`.
- El contexto para el modelo es concatenación de los campos del JSON: `nombre + descripcion + significado + accion + contexto`.
- Retrieval semántico fallback: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- En HF Spaces CPU free tier (~16 GB RAM): todos los modelos cargan sin problema (<2 GB total).
