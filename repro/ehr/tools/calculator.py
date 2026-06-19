"""Stub for the WolframAlpha calculator (no external API available offline).
Handles the simple arithmetic that EHRAgent's Calculate() is used for."""
import re

def WolframAlphaCalculator(formula: str):
    expr = str(formula)
    # keep only arithmetic-safe characters
    if re.fullmatch(r"[0-9eE\.\+\-\*\/\(\)\s,]*", expr):
        try:
            return eval(expr, {"__builtins__": {}}, {})
        except Exception:
            pass
    return f"[calculator stub: cannot evaluate '{formula}']"
