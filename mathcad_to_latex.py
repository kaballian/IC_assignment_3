#!/usr/bin/env python3
"""
mathcad_to_latex.py

Convert Mathcad-style exported token text (prefix S-expr-like with tokens like
(@ID ...), (@SUB ...), (@NTHROOT ...), (@APPLY ...), (@INTEGRAL ...), etc.)
to LaTeX.

Multiplication is rendered EXPLICITLY as \cdot (user chose option 1).
"""

from __future__ import annotations
import re
import sys
from typing import List, Union, Any

# ---------- Utilities ----------
Atom = str
SExpr = Union[Atom, List["SExpr"]]

GREEK = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta",
    "epsilon": r"\epsilon", "zeta": r"\zeta", "eta": r"\eta", "theta": r"\theta",
    "iota": r"\iota", "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho", "sigma": r"\sigma",
    "tau": r"\tau", "upsilon": r"\upsilon", "phi": r"\phi", "chi": r"\chi",
    "psi": r"\psi", "omega": r"\omega", "Δ": r"\Delta", "ω": r"\omega"
}

# ---------- Tokenizer ----------
def tokenize(s: str) -> List[str]:
    """Return list of tokens (parens and atoms). Keep unicode like ∞ intact."""
    # Insert spaces around parentheses
    s2 = s.replace("(", " ( ").replace(")", " ) ")
    # split by whitespace but keep special multi-char tokens intact
    tokens = s2.split()
    return tokens

# ---------- Parser (to nested lists) ----------
def parse_tokens(tokens: List[str]) -> SExpr:
    """Turn tokens list into an S-expression (nested lists)."""
    if not tokens:
        raise ValueError("Empty token stream")

    token = tokens.pop(0)
    if token == "(":
        node: List[SExpr] = []
        while tokens and tokens[0] != ")":
            node.append(parse_tokens(tokens))
        if not tokens:
            raise ValueError("Unmatched '('")
        tokens.pop(0)  # remove ')'
        return node
    elif token == ")":
        raise ValueError("Unexpected ')'")
    else:
        return token

def parse_string(s: str) -> SExpr:
    """Convenience to parse a string into SExpr."""
    tokens = tokenize(s)
    # If the whole expression doesn't start with a single top-level paren,
    # we'll wrap it to make parsing simpler.
    if tokens and tokens[0] != "(":
        tokens = ["("] + tokens + [")"]
    expr = parse_tokens(tokens)
    if tokens:
        # tokens list should be empty or contain only whitespace originally
        pass
    return expr

# ---------- AST -> LaTeX converter ----------
def is_atom(x: SExpr) -> bool:
    return isinstance(x, str)

def atom_text(atom: str) -> str:
    """Return cleaned atom text and map special tokens (greek, infinity, placeholders)."""
    a = atom.strip()
    # remove placeholder markers
    a = a.replace("@PLACEHOLDER", "").strip()

    # map unicode and common spellings for infinity
    if a in ("∞", "infty", "INF", "Infinity"):
        return r"\infty"

    # map unicode pi and common pi spellings
    if a in ("π", "pi", "PI"):
        return r"\pi"

    # map '1j' -> 'j' (complex j) and common numeric tokens
    if a == "1j":
        return "j"

    # clean leftover @ID prefix if any (safety)
    a = re.sub(r'^@ID\s+', '', a)

    # greek mapping fallback (word forms, e.g. 'alpha')
    if a in GREEK:
        return GREEK[a]

    return a



def sexpr_to_latex(node: SExpr) -> str:
    if is_atom(node):
        return atom_text(node)

    # node is a list
    if not node:
        return ""

    head = node[0] if node else ""
    # head might itself be an sexpr (rare), convert to string
    head_str = head if isinstance(head, str) else (head[0] if head else "")

    # Helpers
    def arg(i: int) -> SExpr:
        return node[i] if i < len(node) else ""

    def to(i: int) -> str:
        return sexpr_to_latex(arg(i))

    # Handle common Mathcad tokens
        # Assignment: (:= variable expression)
    if head_str == ":=":
        if len(node) == 3:
            left = sexpr_to_latex(node[1])
            right = sexpr_to_latex(node[2])
            return f"{left} = {right}"
        return "=".join(sexpr_to_latex(c) for c in node[1:])

    # Symbolic evaluation: (@SYM_EVAL expr) -> expr unchanged
    if head_str == "@SYM_EVAL":
        return sexpr_to_latex(node[1]) if len(node) > 1 else ""

    # Ignore stack keyword (@KW_STACK anything) -> remove it
    if head_str == "@KW_STACK":
        return sexpr_to_latex(node[-1])

    # Variables: (@ID name) or (@ID name (@SUB sub))
    if head_str == "@ID":
        name = to(1) if len(node) >= 2 else ""
        
        # convert leading Δ to \Delta
        if name.startswith("Δ"):
            name = r"\Delta " + name[1:]
        
        # handle subscript
        if len(node) >= 3 and isinstance(node[2], list) and node[2][0] == "@SUB":
            sub_tex = sexpr_to_latex(node[2])
            return f"{name}_{{{sub_tex}}}"
        
        return name





    if head_str == "@SUB":
    # Concatenate all children literally, but replace underscores with dashes
        parts = []
        for c in node[1:]:
            if is_atom(c):
                parts.append(c.replace("_", "-"))
            else:
                # recursively convert nested nodes
                parts.append(sexpr_to_latex(c))
        return "".join(parts)



    
    if head_str == "@SUP" or head_str == "@POW":
        return "^{" + to(1) + "}"
    if head_str == "@NEG":
        return "-" + to(1)
    if head_str == "@PLACEHOLDER":
        return ""  # ignore

    if head_str == "@RSCALE":
        # (@RSCALE value unit) → value + "\ " + unit
        value_tex = to(1) if len(node) >= 2 else ""
        unit_tex = ""
        if len(node) >= 3 and isinstance(node[2], list) and node[2][0] == "@LABEL":
            for child in node[2][1:]:
                if isinstance(child, list) and child[0] == "@ID":
                    unit_tex = sexpr_to_latex(child)
                elif is_atom(child):
                    unit_tex = child
        return f"{value_tex}\\ {unit_tex}"



    if head_str == "@LABEL":
        # skip labels (e.g. (@LABEL VARIABLE k)) return inner useful text if present
        # Some labels are structured; try to find an ID or variable token inside
        for child in node[1:]:
            if isinstance(child, list) and child and child[0] == "@ID":
                return sexpr_to_latex(child)
            if isinstance(child, str):
                # e.g. "VARIABLE"
                continue
        return ""

    if head_str == "@PARENS":
        return "(" + "".join(sexpr_to_latex(c) for c in node[1:]) + ")"

    if head_str == "@ABS":
        inner = "".join(sexpr_to_latex(c) for c in node[1:])
        return r"\left| " + inner + r" \right|"

    if head_str == "@APPLY":
        # (@APPLY fname (@ARGS ...))
        fname = to(1)
        args_part = ""
        # find @ARGS
        for child in node[2:]:
            if isinstance(child, list) and child and child[0] == "@ARGS":
                args_part = "".join(sexpr_to_latex(c) for c in child[1:])
                break
        # if args_part empty maybe next atom(s) are the args
        if args_part == "":
            # try second element direct
            if len(node) > 2:
                args_part = "".join(sexpr_to_latex(c) for c in node[2:])
        return f"{fname}({args_part})"

    if head_str == "@ARGS":
        # list of arguments, join with commas
        parts = [sexpr_to_latex(c) for c in node[1:]]
        return ", ".join(p for p in parts if p != "")

    if head_str == "@SCALE":
        # (@SCALE a b) -> a \cdot b (we use explicit dot)
        return " \\cdot ".join(sexpr_to_latex(c) for c in node[1:])

    if head_str == "@NTHROOT":
        inner = "".join(sexpr_to_latex(c) for c in node[1:])
        return r"\sqrt{" + inner + "}"

    # Integral: (@INTEGRAL lower upper expr [var])
    if head_str == "@INTEGRAL":
        # Robust handling for a few Mathcad variants:
        # 1) (@INTEGRAL lower upper expr [var])
        # 2) (@INTEGRAL lower expr var)   -> assume upper = \infty
        # 3) (@INTEGRAL lower expr)       -> assume upper = \infty, var = f
        #
        # We'll inspect the AST shape to decide which form we have.

        n = len(node)
        # helper to detect if a child looks like an expression (list starting with operator or nested)
        def looks_like_expr(ch):
            return isinstance(ch, list)

        if n >= 5:
            # fully explicit form: lower, upper, expr, var
            lower_tex = sexpr_to_latex(node[1])
            upper_tex = sexpr_to_latex(node[2])
            expr_tex  = sexpr_to_latex(node[3])
            var_tex   = sexpr_to_latex(node[4])
            return rf"\int_{{{lower_tex}}}^{{{upper_tex}}} {expr_tex} \, d{var_tex}"

        if n == 4:
            # Ambiguous: could be (low up expr) or (low expr var)
            # Decide by examining node[2]: if node[2] is an atomic bound like '∞' or a simple atom -> treat as upper
            # otherwise if node[2] is a list (an expression) and node[3] likely a var -> treat as (low expr var)
            child2 = node[2]
            child3 = node[3]

            if isinstance(child2, str) and child2.strip() in ("∞", "infty", "INF", "Infinity", r"\infty"):
                # form: (low upper expr) but expr is node[3] and missing var -> assume var = f
                lower_tex = sexpr_to_latex(node[1])
                upper_tex = sexpr_to_latex(node[2])
                expr_tex  = sexpr_to_latex(node[3])
                return rf"\int_{{{lower_tex}}}^{{{upper_tex}}} {expr_tex} \, df"

            if isinstance(child2, list) and (isinstance(child3, str) and len(child3) == 1):
                # looks like: (low expr var) -> upper = \infty
                lower_tex = sexpr_to_latex(node[1])
                upper_tex = r"\infty"
                expr_tex  = sexpr_to_latex(node[2])
                var_tex   = sexpr_to_latex(node[3])
                return rf"\int_{{{lower_tex}}}^{{{upper_tex}}} {expr_tex} \, d{var_tex}"

            # fallback: assume form (low upper expr)
            lower_tex = sexpr_to_latex(node[1])
            upper_tex = sexpr_to_latex(node[2])
            expr_tex  = sexpr_to_latex(node[3])
            return rf"\int_{{{lower_tex}}}^{{{upper_tex}}} {expr_tex} \, df"

        if n == 3:
            # form: (@INTEGRAL lower expr) -> upper = \infty, var = f
            lower_tex = sexpr_to_latex(node[1])
            upper_tex = r"\infty"
            expr_tex  = sexpr_to_latex(node[2])
            return rf"\int_{{{lower_tex}}}^{{{upper_tex}}} {expr_tex} \, df"

        # fallback
        return r"\int " + " ".join(sexpr_to_latex(c) for c in node[1:])





    if head_str == "@SUM":
        # Pattern: (@SUM (@IS var start) end expr)
        if len(node) >= 4:
            # left: @IS var start
            is_node = node[1]
            if isinstance(is_node, list) and is_node[0] == "@IS":
                var = sexpr_to_latex(is_node[1])
                start = sexpr_to_latex(is_node[2])
                end = sexpr_to_latex(node[2])
                expr_in = sexpr_to_latex(node[3])
                return rf"\sum_{{{var}={start}}}^{{{end}}} {expr_in}"
        # fallback: just concatenate
        parts = [sexpr_to_latex(c) for c in node[1:]]
        return r"\sum " + " ".join(parts)


    if head_str == "@DERIV":
        # (@DERIV x expr) -> d/dx (expr)
        if len(node) >= 3:
            var = sexpr_to_latex(node[1])
            expr_in = sexpr_to_latex(node[2])
            return rf"\frac{{d}}{{d{var}}}\left({expr_in}\right)"
        return r"\frac{d}{dx}"

    if head_str == "@LIMIT" or head_str == "@LIM":
        # (@LIMIT var value expr) -> \lim_{var \to value} expr
        if len(node) >= 4:
            var = sexpr_to_latex(node[1])
            val = sexpr_to_latex(node[2])
            expr_in = sexpr_to_latex(node[3])
            return rf"\lim_{{{var} \to {val}}} {expr_in}"
        return r"\lim " + " ".join(sexpr_to_latex(c) for c in node[1:])
    if head_str == "@EQ":
        # (@EQ left right) -> left = right
        if len(node) >= 3:
            left = sexpr_to_latex(node[1])
            right = sexpr_to_latex(node[2])
            return f"{left} = {right}"
        return "=".join(sexpr_to_latex(c) for c in node[1:])
    

    # Arithmetic prefix forms: (^ + - * /)
    if head_str == "^" or head_str == "(^)":
        if len(node) >= 3:
            base = sexpr_to_latex(node[1])
            power = sexpr_to_latex(node[2])
            return rf"{base}^{{{power}}}"
    if head_str == "+" or head_str == "(+)":
        parts = [sexpr_to_latex(c) for c in node[1:]]
        return " + ".join(parts)
    if head_str == "-" or head_str == "(-)":
        if len(node) == 2:
            return "-" + sexpr_to_latex(node[1])
        parts = [sexpr_to_latex(c) for c in node[1:]]
        return " - ".join(parts)
    if head_str == "*" or head_str == "(* )" or head_str == "(@MUL)" or head_str == "(@MULTIPLY)":
        # join with explicit \cdot
        parts = [sexpr_to_latex(c) for c in node[1:]]
        # filter empty
        parts = [p for p in parts if p]
        return r" \cdot ".join(parts)
    if head_str == "/":
        if len(node) >= 3:
            num = sexpr_to_latex(node[1])
            den = sexpr_to_latex(node[2])
            return rf"\frac{{{num}}}{{{den}}}"
    # fallback: unrecognized head -> render as concatenation of children
    return "".join(sexpr_to_latex(c) for c in node)

# ---------- Top-level conversion function ----------
def convert_mathcad_text_to_latex(s: str) -> str:
    """
    Accepts a string containing Mathcad-style tokens and returns the LaTeX string.
    """
    # Quick cleanups to help tokenizer:
    # - ensure parentheses are spaced
    # - remove known no-op wrappers and placeholders
    s_clean = s.replace("@PLACEHOLDER", "")
    # Sometimes the export uses unicode ∞ or unusual tokens; keep them as atoms.
    # Parse to SExpr
    try:
        ast = parse_string(s_clean)
    except Exception as e:
        # fallback: try to find top-level parentheses and parse those
        raise RuntimeError(f"Parsing error: {e}")

    latex = sexpr_to_latex(ast)
    # Some cosmetic fixes:
    latex = re.sub(r"\s+", " ", latex).strip()
    # fix repeated \cdot spacing
    latex = latex.replace(" \\cdot  \\cdot ", " \\cdot ")
    return latex

# ---------- If module run as script ----------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # join args as input text
        inp = " ".join(sys.argv[1:])
    else:
        # read from stdin
        print("Enter Mathcad-style token text (end with EOF / Ctrl-D):")
        inp = sys.stdin.read()

    if not inp.strip():
        print("No input provided.")
        sys.exit(1)

    out = convert_mathcad_text_to_latex(inp)
    print("\nLaTeX output:\n")
    print(out)
