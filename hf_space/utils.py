import numpy as np
from PIL import Image

IMG_SIZE = 32
HF_MODEL_REPO = "Medina07/traffic-signs-cnn"
HF_MODEL_FILENAME = "cnn_gtsrb_v1.h5"


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    arr = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def load_cnn():
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.models import load_model

    path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    return load_model(path)
