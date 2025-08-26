
import argparse, os, numpy as np, pandas as pd
from src.preprocess import tokenize_code, simple_identifier_normalize, tokens_to_string
from src.embeddings import CodeEmbedder

def compute(data_path, out_dir, model_name, batch_size):
    df = pd.read_csv(data_path)
    textsA, textsB = [], []
    for a,b in zip(df['fileA'], df['fileB']):
        textsA.append(tokens_to_string(simple_identifier_normalize(tokenize_code(a))))
        textsB.append(tokens_to_string(simple_identifier_normalize(tokenize_code(b))))
    os.makedirs(out_dir, exist_ok=True)
    ce = CodeEmbedder(model_name=model_name)
    print("Computing embeddings for A...")
    embA = ce.embed(textsA, batch_size=batch_size)
    print("Computing embeddings for B...")
    embB = ce.embed(textsB, batch_size=batch_size)
    np.save(os.path.join(out_dir, 'embA.npy'), embA)
    np.save(os.path.join(out_dir, 'embB.npy'), embB)
    meta = {'model_name': model_name, 'shapeA': embA.shape, 'shapeB': embB.shape}
    with open(os.path.join(out_dir, 'emb_meta.json'), 'w') as f:
        import json
        json.dump(meta, f)
    print(f"Saved embeddings to {out_dir}/embA.npy and embB.npy")
