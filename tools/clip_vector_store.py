import os
import numpy as np
import faiss
import open_clip
from PIL import Image
import torch

class CLIPVectorStore:
    def __init__(self, index_path):
        self.device = "cpu"
        # Use the -quickgelu model to avoid the warning
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        self.index_path = index_path
        self.index = None
        self.embeddings = []
        self.metadatas = []
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(512)

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def add(self, embedding, metadata):
        self.index.add(np.array([embedding]).astype(np.float32))
        self.metadatas.append(metadata)
        self.save()

    def embed_text(self, text):
        with torch.no_grad():
            tokens = self.tokenizer([text])
            tokens = tokens.to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb.cpu().numpy()[0]
        return emb

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image)
            emb = emb.cpu().numpy()[0]
        return emb