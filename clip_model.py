

from langchain_core.embeddings import Embeddings
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch






class CLIPImageEmbeddings(Embeddings):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def embed_documents(self, texts):
        # For text-based documents (not used here)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs).cpu().numpy()[0]
        return embedding.tolist()

    def embed_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs).cpu().numpy()[0]
        return embedding.tolist()




if __name__ == "__main__":

    print()



