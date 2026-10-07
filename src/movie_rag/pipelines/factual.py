import re

import numpy as np
import pandas as pd

from movie_rag.safety.validators import SAFE_BUILTINS, UnsafeCodeError, validate_code

MAX_RESULT_ROWS = 25


def describe_data(rich_movies) -> str:
    """Facts about the actual data, so the LLM uses real values (e.g. certificate codes) and ranges."""
    if rich_movies is None or "Certificate" not in rich_movies:
        return ""
    certs = ", ".join(rich_movies["Certificate"].dropna().astype(str).value_counts().index[:12])
    lines = [f"- {len(rich_movies)} movies; Certificate values: {certs}"]
    for col in ["Year", "Rating", "Votes", "Gross(Million)"]:
        if col in rich_movies:
            lines.append(f"- {col} ranges {rich_movies[col].min():,.10g} to {rich_movies[col].max():,.10g}"
                         f" ({rich_movies[col].isna().sum()} missing)")
    return "\nData facts:\n" + "\n".join(lines) + "\n"


def build_factual_prompt(query: str, rich_movies=None) -> str:
    return f"""Return ONLY one Python code block (and nothing else).

Rules:
- Use ONLY the pandas DataFrame `rich_movies` (pandas is NOT imported, do not use `pd`)
- Do NOT print
- Do NOT explain
- Output ONE LINE ONLY
- Assign the final answer to variable `result`
- Do NOT use ellipsis (...), lambda, loops, f-strings or inplace=
- When listing movies, return a DataFrame with Title, Year and every column used to filter or sort (e.g. Gross(Million) for "highest grossing")

Columns in rich_movies:
- Title (str), Year (int), Runtime (minutes, float), Rating (IMDB 0-10, float)
- Genres (str, comma-separated, e.g. "Action, Adventure, Sci-Fi") -> filter with .str.contains("Action", case=False, na=False)
- Certificate (str), Metascore (0-100, float, may be NaN)
- Votes (int), Gross(Million) (US gross in millions of dollars, float, NaN when unknown)
- Director (str), Stars (str, comma-separated actor names)
- Summary (str, one-sentence plot)
{describe_data(rich_movies)}
Question:
{query}

Output format:
```python
result = <single pandas expression using rich_movies>
```
""".strip()


def extract_code_from_response(response_text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise ValueError("No ```python``` block found from LLM.")
    return blocks[-1].strip()


def run_factual_code(code: str, rich_movies):
    validate_code(code)
    env = {"__builtins__": SAFE_BUILTINS, "rich_movies": rich_movies}
    exec(compile(code, "<factual>", "exec"), env)
    return env.get("result", None)


def to_display(result):
    """Convert a pandas/numpy result into (kind, value) that a UI or JSON API can show."""
    if isinstance(result, pd.Series):
        result = result.to_frame(name=result.name or "value").reset_index()
    if isinstance(result, pd.DataFrame):
        df = result.head(MAX_RESULT_ROWS).copy()
        if "text" in df.columns:
            df = df.drop(columns=["text"])
        return "table", df
    if isinstance(result, np.generic):
        result = result.item()
    if isinstance(result, float):
        return "scalar", round(result, 3)
    if isinstance(result, np.ndarray):
        return "scalar", result[:MAX_RESULT_ROWS].tolist()
    return "scalar", result


def factual_pipeline(query: str, *, rich_movies, call_llm_fn, max_attempts: int = 2):
    prompt = build_factual_prompt(query, rich_movies)
    last_error = None
    code = None

    for attempt in range(max_attempts):
        attempt_prompt = prompt
        if last_error:
            attempt_prompt += (
                f"\n\nYour previous answer was:\n{code}\nIt failed with: {last_error}\n"
                "Return a corrected single line that follows ALL rules."
            )
        raw = call_llm_fn(attempt_prompt)  # API errors (no key, rate limit) propagate, no retry
        try:
            code = extract_code_from_response(raw)
            result = run_factual_code(code, rich_movies)
            kind, value = to_display(result)
            return {"query": query, "generated_code": code, "result": value, "result_kind": kind}
        except UnsafeCodeError as e:
            last_error = f"rejected by safety validator: {e}"
        except Exception as e:  # bad column, wrong dtype, etc. -> let the LLM fix it once
            last_error = f"{type(e).__name__}: {e}"

    raise RuntimeError(f"Could not answer factual query after {max_attempts} attempts. Last error: {last_error}")
