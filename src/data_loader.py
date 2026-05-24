from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import IMG_SIZE, VALIDATION_SPLIT


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


def load_gtsrb(train_dir: Path, test_csv: Path) -> Tuple:
    """
    Carga el dataset GTSRB completo.
    train_dir: carpeta Train/ con 43 subcarpetas (0-42).
    test_csv: ruta al archivo Test.csv del GTSRB.
    Retorna: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    images, labels = [], []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_id = int(cls_dir.name)
        for img_path in cls_dir.glob("*.png"):
            images.append(_load_image(img_path))
            labels.append(cls_id)

    X = np.array(images)
    y = np.array(labels, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    df_test = pd.read_csv(test_csv)
    test_images = [_load_image(test_csv.parent / row["Path"]) for _, row in df_test.iterrows()]
    X_test = np.array(test_images)
    y_test = df_test["ClassId"].to_numpy(dtype=np.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))
