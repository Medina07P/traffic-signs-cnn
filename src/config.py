from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
KB_PATH = DATA_DIR / "knowledge_base" / "traffic_signs_es.json"

IMG_SIZE = 32
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

HF_MODEL_REPO = "jarolmedina41/traffic-signs-cnn"
HF_MODEL_FILENAME = "cnn_gtsrb_v1.h5"
QA_MODEL_NAME = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
RETRIEVER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QA_CONFIDENCE_THRESHOLD = 0.15

LABEL_MAP = {
    0:  "Límite de velocidad 20 km/h",
    1:  "Límite de velocidad 30 km/h",
    2:  "Límite de velocidad 50 km/h",
    3:  "Límite de velocidad 60 km/h",
    4:  "Límite de velocidad 70 km/h",
    5:  "Límite de velocidad 80 km/h",
    6:  "Fin de límite de velocidad 80 km/h",
    7:  "Límite de velocidad 100 km/h",
    8:  "Límite de velocidad 120 km/h",
    9:  "Prohibido adelantar",
    10: "Prohibido adelantar vehículos de más de 3,5 toneladas",
    11: "Derecho de paso en la próxima intersección",
    12: "Vía preferencial",
    13: "Ceda el paso",
    14: "Alto",
    15: "Circulación prohibida",
    16: "Prohibido vehículos de más de 3,5 toneladas",
    17: "Prohibido el paso",
    18: "Peligro general",
    19: "Curva peligrosa a la izquierda",
    20: "Curva peligrosa a la derecha",
    21: "Doble curva",
    22: "Resalto o badén",
    23: "Pavimento resbaladizo",
    24: "Vía estrecha por la derecha",
    25: "Obras en la vía",
    26: "Semáforo adelante",
    27: "Peatones",
    28: "Paso de escolares",
    29: "Cruce de ciclistas",
    30: "Peligro de hielo o nieve",
    31: "Cruce de animales silvestres",
    32: "Fin de todas las restricciones",
    33: "Gire a la derecha",
    34: "Gire a la izquierda",
    35: "Siga adelante",
    36: "Adelante o gire a la derecha",
    37: "Adelante o gire a la izquierda",
    38: "Mantenga la derecha",
    39: "Mantenga la izquierda",
    40: "Glorieta obligatoria",
    41: "Fin de prohibición de adelantar",
    42: "Fin de prohibición de adelantar para vehículos de más de 3,5 toneladas",
}
