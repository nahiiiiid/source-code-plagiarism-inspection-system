from src.preprocess import remove_comments, tokenize_code, simple_identifier_normalize, line_level_similarity

def test_remove_comments():
    s = "# hello\n def f():\n    pass //inline"
    out = remove_comments(s)
    assert 'hello' not in out

def test_tokenize_and_normalize():
    code = "def add(a, b):\n    return a + b"
    toks = tokenize_code(code)
    norm = simple_identifier_normalize(toks)
    assert 'def' in toks and (len(norm)>0)
