# Detección de Señales de Tránsito para Conducción Autónoma

Proyecto Final — Curso de Inteligencia Artificial, Tercer Corte  
Fundación Universitaria de Popayán (FUP)

## Descripción

Sistema de visión computacional que detecta y clasifica señales de tránsito del dataset GTSRB (43 clases) en tiempo real, complementado con un agente conversacional que responde preguntas sobre las señales usando Transformers en español.

**App desplegada:** https://huggingface.co/spaces/jarolmedina41/traffic-signs-app

## Componentes

| Componente | Tecnología | Descripción |
|---|---|---|
| Clasificador CNN | TensorFlow/Keras | 43 clases GTSRB, entrada 32×32×3, ~93% accuracy |
| Detección tiempo real | OpenCV | Captura webcam, clasificación frame a frame |
| Agente Q&A | BERT español (HF Transformers) | Responde preguntas sobre señales |
| Interfaz web | Gradio | App responsiva, accesible desde móvil |
| Despliegue | Hugging Face Spaces | Acceso gratuito sin instalación |

## Reproducir el entrenamiento

1. Ir a [Kaggle](https://www.kaggle.com) y crear una cuenta.
2. Crear nuevo Notebook → Settings → Accelerator: **GPU T4**.
3. Agregar dataset: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`.
4. Subir y ejecutar `notebooks/01_train_cnn_kaggle.ipynb`.
5. Descargar `cnn_gtsrb_v1.h5` del panel Output.
6. Subir el modelo a HF Hub:
   ```bash
   huggingface-cli login
   huggingface-cli upload jarolmedina41/traffic-signs-cnn models/cnn_gtsrb_v1.h5
   ```

## Estructura del proyecto

```
PROYECTO/
├── src/                    # Módulos Python reutilizables
├── notebooks/              # Notebooks de entrenamiento y EDA
├── hf_space/               # App Gradio (desplegada en HF Spaces)
├── data/knowledge_base/    # Base de conocimiento de las 43 señales en español
├── models/                 # Modelo entrenado (no versionado en git)
└── reports/figures/        # Gráficas de métricas
```

## Uso local

```bash
pip install -r requirements-local.txt
cd hf_space
python app.py
# Abrir http://localhost:7860
```

## Dataset

GTSRB — German Traffic Sign Recognition Benchmark  
- 43 clases, ~50.000 imágenes de entrenamiento  
- Distribución desbalanceada (200–2000 imágenes por clase)  
- Descarga: `kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`
