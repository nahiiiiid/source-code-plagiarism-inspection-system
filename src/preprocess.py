import re
from typing import List, Tuple
import ast

# Remove comments (simple)
SINGLE_LINE_COMMENT = re.compile(r"#.*?$|//.*?$", re.MULTILINE)
MULTI_LINE_COMMENT = re.compile(r"/\*.*?\*/|'''.*?'''|\"\"\".*?\"\"\"", re.DOTALL)

KEYWORDS = set([
    'def','return','if','else','for','while','import','from','class','try','except','with','as','in','and','or','not','True','False','None'
])

def remove_comments(code: str) -> str:
    code = re.sub(MULTI_LINE_COMMENT, ' ', code)
    code = re.sub(SINGLE_LINE_COMMENT, ' ', code)
    return code

def normalize_whitespace(code: str) -> str:
    code = re.sub(r"\r\n|\r", "\n", code)
    code = re.sub(r"[ \t]+", ' ', code)
    code = re.sub(r"\n\s*\n+", '\n', code)
    return code.strip()

# AST-based normalization for Python: replace identifiers with generic tokens while preserving structure
def ast_normalize_python(code: str) -> str:
    try:
        tree = ast.parse(code)
    except Exception:
        # fallback
        return None
    # mapping for names
    mapping = {}
    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
                if node.id not in mapping and node.id not in KEYWORDS:
                    mapping[node.id] = f"ID_{len(mapping)+1}"
                if node.id in mapping:
                    return ast.copy_location(ast.Name(id=mapping[node.id], ctx=node.ctx), node)
            return node
        def visit_arg(self, node):
            if node.arg not in mapping and node.arg not in KEYWORDS:
                mapping[node.arg] = f"ID_{len(mapping)+1}"
            if node.arg in mapping:
                node.arg = mapping[node.arg]
            return node
    renamer = Renamer()
    new_tree = renamer.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        import astunparse
        return astunparse.unparse(new_tree)
    except Exception:
        return None

def tokenize_code(code: str) -> List[str]:
    code = remove_comments(code)
    code = normalize_whitespace(code)
    # try AST normalize for Python
    norm = ast_normalize_python(code)
    if norm:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\+\+|--|[{}()\[\];,.:+\-/*%<>]", norm)
        return tokens
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\+\+|--|[{}()\[\];,.:+\-/*%<>]", code)
    return tokens

def simple_identifier_normalize(tokens: List[str]) -> List[str]:
    mapping = {}
    next_idx = 1
    out = []
    for t in tokens:
        if t in KEYWORDS or re.fullmatch(r"\d+", t):
            out.append(t)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
            if t not in mapping:
                mapping[t] = f"ID_{next_idx}"
                next_idx += 1
            out.append(mapping[t])
        else:
            out.append(t)
    return out

def tokens_to_string(tokens: List[str]) -> str:
    return ' '.join(tokens)

def code_lines(code: str) -> List[str]:
    return [l for l in code.splitlines() if l.strip()!='']

def line_level_similarity(codeA: str, codeB: str) -> List[tuple]:
    import difflib
    linesA = code_lines(codeA)
    linesB = code_lines(codeB)
    matches = []
    for i,a in enumerate(linesA):
        best = (None, 0.0, None)
        for j,b in enumerate(linesB):
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r>best[1]:
                best = (j, r, b)
        if best[0] is not None and best[1]>0.5:
            matches.append((i, best[0], best[1], a, best[2]))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches
