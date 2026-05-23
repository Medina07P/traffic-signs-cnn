from pathlib import Path

import numpy as np
from PIL import Image

from src.config import IMG_SIZE, HF_MODEL_REPO, HF_MODEL_FILENAME, MODELS_DIR


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    arr = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def load_model_from_hub():
    """Descarga cnn_gtsrb_v1.h5 desde HF Hub y lo carga como modelo Keras."""
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.models import load_model

    local_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    return load_model(local_path)


def load_model_local(path: Path = None):
    from tensorflow.keras.models import load_model as keras_load

    model_path = path or (MODELS_DIR / HF_MODEL_FILENAME)
    return keras_load(str(model_path))
