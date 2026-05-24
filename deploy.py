"""
Script de despliegue a Hugging Face.
Ejecutar DESPUÉS de: huggingface-cli login
"""
from pathlib import Path
from huggingface_hub import HfApi

MODEL_REPO  = "jarolmedina41/traffic-signs-cnn"
SPACE_REPO  = "jarolmedina41/traffic-signs-app"
MODEL_FILE  = Path("models/cnn_gtsrb_v1.h5")
SPACE_DIR   = Path("hf_space")

api = HfApi()

# ── 1. Subir modelo al Hub ───────────────────────────────────────────────────
print(f"\n[1/2] Subiendo modelo a {MODEL_REPO} ...")
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=str(MODEL_FILE),
    path_in_repo=MODEL_FILE.name,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"     OK → https://huggingface.co/{MODEL_REPO}")

# ── 2. Subir Space ───────────────────────────────────────────────────────────
print(f"\n[2/2] Subiendo Space a {SPACE_REPO} ...")
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(
    folder_path=str(SPACE_DIR),
    repo_id=SPACE_REPO,
    repo_type="space",
)
print(f"     OK → https://huggingface.co/spaces/{SPACE_REPO}")
print("\nDespliegue completado.")
