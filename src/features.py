import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tokens_to_string(tokens):
    return ' '.join(tokens)

class TfidfCodeVectorizer:
    def __init__(self, ngram_range=(1,3), max_features=30000):
        self.vec = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, max_features=max_features)

    def fit(self, code_texts):
        self.vec.fit(code_texts)
        return self

    def transform(self, code_texts):
        return self.vec.transform(code_texts)

    def save(self, path):
        joblib.dump(self.vec, path)

    def load(self, path):
        self.vec = joblib.load(path)
        return self

def cosine_sim_from_vectors(v1, v2):
    return float(cosine_similarity(v1, v2)[0,0])

def ngram_set(tokens, n):
    s = set()
    for i in range(len(tokens)-n+1):
        s.add(' '.join(tokens[i:i+n]))
    return s

def jaccard_ngrams(tokensA, tokensB, n=3):
    a = ngram_set(tokensA, n)
    b = ngram_set(tokensB, n)
    if not a and not b: return 0.0
    return len(a & b) / len(a | b)

def lcs_ratio(s1, s2):
    m, n = len(s1), len(s2)
    if m==0 or n==0: return 0.0
    dp = [0]*(n+1)
    for i in range(1, m+1):
        prev = 0
        for j in range(1, n+1):
            tmp = dp[j]
            if s1[i-1]==s2[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    lcs = dp[-1]
    return lcs / max(m,n)
