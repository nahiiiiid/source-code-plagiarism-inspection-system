#!/usr/bin/env python3
import argparse, os, pandas as pd, itertools, csv
from src.preprocess import remove_comments, tokenize_code, simple_identifier_normalize, line_level_similarity, tokens_to_string
from src.features import TfidfCodeVectorizer, cosine_sim_from_vectors
from src.embeddings import CodeEmbedder
import numpy as np

def read_code(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def scan_folder(folder, threshold=0.8, backend='tfidf', codebert_model='microsoft/codebert-base'):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    names = [os.path.basename(f) for f in files]
    pairs = list(itertools.combinations(range(len(files)), 2))
    rows = []
    codes = [read_code(f) for f in files]
    toks = [simple_identifier_normalize(tokenize_code(c)) for c in codes]
    strs = [tokens_to_string(t) for t in toks]
    tfidf = TfidfCodeVectorizer()
    tfidf.fit(strs)
    emb = None
    if backend=='codebert':
        ce = CodeEmbedder(model_name=codebert_model)
        emb = ce.embed(strs, batch_size=8)
    for i,j in pairs:
        sA, sB = codes[i], codes[j]
        vA = tfidf.transform([strs[i]])
        vB = tfidf.transform([strs[j]])
        cos = cosine_sim_from_vectors(vA, vB)
        from src.features import jaccard_ngrams, lcs_ratio
        jacc = jaccard_ngrams(toks[i], toks[j], n=3)
        lcs = lcs_ratio(strs[i].split(), strs[j].split())
        score = 0.5*cos + 0.3*jacc + 0.2*lcs
        if emb is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            emb_cos = float(cosine_similarity(emb[i:i+1], emb[j:j+1])[0,0])
            score = 0.6*emb_cos + 0.25*cos + 0.15*jacc
        if score >= threshold:
            matches = line_level_similarity(sA, sB)
            rows.append({
                'fileA': names[i],
                'fileB': names[j],
                'score': score,
                'n_matches': len(matches),
                'top_matches': repr(matches[:5])
            })
    return rows

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--out', default='report.csv')
    p.add_argument('--threshold', type=float, default=0.8)
    p.add_argument('--backend', choices=['tfidf','codebert'], default='tfidf')
    args = p.parse_args()
    rows = scan_folder(args.folder, threshold=args.threshold, backend=args.backend)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('Wrote', args.out)
