import pytest

from movie_rag.pipelines.factual import run_factual_code
from movie_rag.safety.validators import UnsafeCodeError, validate_code

SAFE = [
    'result = rich_movies[rich_movies["Genres"].str.contains("Action", na=False) & (rich_movies["Year"] > 2010)]["Rating"].mean()',
    'result = rich_movies.nlargest(5, "Gross(Million)")[["Title", "Year", "Gross(Million)"]]',
    'result = len(rich_movies[rich_movies["Rating"] > 8.5])',
    'result = rich_movies[rich_movies["Rating"] > 8].groupby("Director").size().sort_values(ascending=False).head(1)',
    'result = rich_movies["Gross(Million)"].sum() * 1e6',
    'result = round(rich_movies["Runtime"].mean(), 1)',
    'result = rich_movies.loc[rich_movies["Votes"].idxmax(), "Title"]',
]

UNSAFE = [
    # sandbox escapes
    'result = rich_movies.__class__.__init__.__globals__',
    'result = rich_movies.__class__.__mro__[-1].__subclasses__()',
    'result = __import__("os").system("id")',
    'result = rich_movies.pipe(eval, "1")',
    'result = rich_movies.apply(exec)',
    'result = getattr(rich_movies, "to_csv")("/tmp/x")',
    'result = rich_movies.query("@__builtins__")',
    'result = rich_movies.eval("1+1")',
    'result = open("/etc/passwd").read()',
    # filesystem writes / mutation of the shared DataFrame
    'result = rich_movies.to_csv("/tmp/x.csv")',
    'result = rich_movies.to_pickle("/tmp/x.pkl")',
    'result = rich_movies.dropna(inplace=True)',
    'result = rich_movies.values.fill(0)',
    # memory bombs
    'result = rich_movies["Title"] * 1000000000',
    'result = rich_movies["Title"] * 100 * 100 * 100',
    'result = "a" * len(rich_movies)',
    'result = 10 ** 100000000',
    'result = rich_movies["Title"].str.pad(1000000000)',
    'result = rich_movies["Title"].str.repeat(1000000)',
    # structure rules
    'x = rich_movies',
    'result = 1',
    'result = rich_movies\nimport os',
    'result = rich_movies; import os',
    'result = [m for m in rich_movies]',
    'result = (lambda: rich_movies)()',
    'result = f"{rich_movies}"',
    'result = rich_movies.head(**{"n": 1})',
    'result = ...',
    'result = rich_movies.head(',
]


@pytest.mark.parametrize("code", SAFE)
def test_safe_code_passes_and_runs(code, movies):
    validate_code(code)
    run_factual_code(code, movies)


@pytest.mark.parametrize("code", UNSAFE)
def test_unsafe_code_rejected(code):
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


def test_exec_has_no_real_builtins(movies):
    # even if the validator were bypassed, exec gets only the safe builtin subset
    with pytest.raises(UnsafeCodeError):
        run_factual_code('result = __import__("os")', movies)


def test_shared_dataframe_not_mutated(movies):
    before = movies.copy()
    run_factual_code('result = rich_movies.sort_values("Rating").reset_index(drop=True)', movies)
    assert movies.equals(before)
