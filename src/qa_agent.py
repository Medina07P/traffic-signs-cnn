import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import KB_PATH, QA_MODEL_NAME, RETRIEVER_MODEL, QA_CONFIDENCE_THRESHOLD, LABEL_MAP


class TrafficSignAgent:
    def __init__(self, kb_path: Path = KB_PATH):
        from transformers import pipeline
        from sentence_transformers import SentenceTransformer

        with open(kb_path, encoding="utf-8") as f:
            self.kb: dict = json.load(f)

        self.qa_pipe = pipeline("question-answering", model=QA_MODEL_NAME)
        self.retriever = SentenceTransformer(RETRIEVER_MODEL)

        self._kb_keys = list(self.kb.keys())
        self._kb_texts = [
            f"{v['nombre']}. {v['descripcion']}. {v['significado']}."
            for v in self.kb.values()
        ]
        self._kb_embeddings = self.retriever.encode(self._kb_texts, convert_to_tensor=True)

    def _build_context(self, entry: dict) -> str:
        return (
            f"{entry['nombre']}. {entry['descripcion']}. "
            f"{entry['significado']}. {entry['accion']}. {entry['contexto']}"
        )

    def _retrieve_best(self, question: str) -> str:
        from sentence_transformers import util

        q_emb = self.retriever.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self._kb_embeddings)[0]
        best_idx = int(scores.argmax())
        return self._kb_keys[best_idx]

    def answer(self, question: str, current_class_id: Optional[int] = None) -> dict:
        if current_class_id is not None:
            class_key = str(current_class_id)
        else:
            class_key = self._retrieve_best(question)

        entry = self.kb[class_key]
        context = self._build_context(entry)

        result = self.qa_pipe(question=question, context=context)
        score = result["score"]

        if score < QA_CONFIDENCE_THRESHOLD:
            answer_text = "No tengo información suficiente sobre esa señal."
        else:
            answer_text = result["answer"]

        return {
            "respuesta": answer_text,
            "confianza": round(score, 4),
            "fuente_clase": entry["nombre"],
        }
