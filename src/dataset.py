\
import argparse
import random
import pandas as pd
from pathlib import Path
from src.preprocess import tokenize_code, simple_identifier_normalize

def make_small_variation(code: str) -> str:
    tokens = tokenize_code(code)
    norm = simple_identifier_normalize(tokens)
    return ' '.join(norm)

def generate_synthetic(out_path: str, n_samples=1000):
    examples = [
        "def add(a, b):\\n    return a + b\\n",
        "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)\\n",
        "class Counter:\\n    def __init__(self):\\n        self.c = 0\\n    def inc(self):\\n        self.c += 1\\n",
    ]

    rows = []
    for _ in range(n_samples):
        a = random.choice(examples)
        if random.random() < 0.5:
            b = make_small_variation(a)
            label = 1
        else:
            b = random.choice(examples)
            label = 0 if b != a else 1
        rows.append({'fileA': a, 'fileB': b, 'label': label})

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} pairs to {out_path}")
