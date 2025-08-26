import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.features import TfidfCodeVectorizer, cosine_sim_from_vectors, tokens_to_string, jaccard_ngrams, lcs_ratio
from src.preprocess import tokenize_code, simple_identifier_normalize
import os

try:
    from src.embeddings import CodeEmbedder
    CODEBERT_AVAILABLE = True
except Exception:
    CODEBERT_AVAILABLE = False

def featurize_pair(codeA, codeB, tfidf=None, embA=None, embB=None):
    tA = simple_identifier_normalize(tokenize_code(codeA))
    tB = simple_identifier_normalize(tokenize_code(codeB))
    sA, sB = tokens_to_string(tA), tokens_to_string(tB)
    feats = []
    if tfidf is not None:
        vA = tfidf.transform([sA])
        vB = tfidf.transform([sB])
        feats.append(cosine_sim_from_vectors(vA, vB))
    else:
        feats.append(0.0)
    feats.append(jaccard_ngrams(tA, tB, n=3))
    feats.append(lcs_ratio(sA.split(), sB.split()))
    if embA is not None and embB is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        feats.append(float(cosine_similarity(embA.reshape(1,-1), embB.reshape(1,-1))[0,0]))
        feats.append(float(np.linalg.norm(embA - embB)))
    return np.array(feats, dtype=float)

def build_feature_matrix(df, tfidf=None, embA_arr=None, embB_arr=None):
    X = []
    for idx, (a,b) in enumerate(zip(df['fileA'], df['fileB'])):
        embA = embA_arr[idx] if embA_arr is not None else None
        embB = embB_arr[idx] if embB_arr is not None else None
        X.append(featurize_pair(a,b,tfidf=tfidf, embA=embA, embB=embB))
    return np.vstack(X)

def train(args):
    df = pd.read_csv(args.data)
    y = df['label'].values

    # TF-IDF baseline
    corpus = []
    for c in pd.concat([df['fileA'], df['fileB']]).unique():
        toks = simple_identifier_normalize(tokenize_code(c))
        corpus.append(tokens_to_string(toks))
    tfidf = TfidfCodeVectorizer()
    tfidf.fit(corpus)
    joblib.dump(tfidf.vec, args.out_vec.replace('.pkl','_tfidf.pkl'))

    embA_arr = embB_arr = None
    if args.backend == 'codebert':
        emb_dir = args.emb_dir
        embA_path = os.path.join(emb_dir, 'embA.npy')
        embB_path = os.path.join(emb_dir, 'embB.npy')
        if os.path.exists(embA_path) and os.path.exists(embB_path):
            embA_arr = np.load(embA_path)
            embB_arr = np.load(embB_path)
            print('Loaded precomputed embeddings.')
        else:
            if not CODEBERT_AVAILABLE:
                raise RuntimeError('CodeBERT not available; run src/compute_embeddings.py on Colab or install transformers+torch.')
            embedder = CodeEmbedder(model_name=args.codebert_model)
            corpusA = [tokens_to_string(simple_identifier_normalize(tokenize_code(x))) for x in df['fileA']]
            corpusB = [tokens_to_string(simple_identifier_normalize(tokenize_code(x))) for x in df['fileB']]
            embA_arr = embedder.embed(corpusA, batch_size=args.batch_size)
            embB_arr = embedder.embed(corpusB, batch_size=args.batch_size)
            os.makedirs(args.emb_dir, exist_ok=True)
            np.save(os.path.join(args.emb_dir,'embA.npy'), embA_arr)
            np.save(os.path.join(args.emb_dir,'embB.npy'), embB_arr)
            print('Saved embeddings to', args.emb_dir)

    X = build_feature_matrix(df, tfidf=tfidf, embA_arr=embA_arr, embB_arr=embB_arr)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    print('Val ROC-AUC:', auc)
    joblib.dump(clf, args.out_model)
    print('Saved model to', args.out_model)
