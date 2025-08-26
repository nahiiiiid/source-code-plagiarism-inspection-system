from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class CodeEmbedder:
    def __init__(self, model_name: str = 'microsoft/codebert-base'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def embed(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_emb = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1)
                summed = (last_hidden * mask).sum(1)
                lengths = mask.sum(1).clamp(min=1e-9)
                emb = (summed / lengths).cpu().numpy()
                all_emb.append(emb)
        return np.vstack(all_emb)
