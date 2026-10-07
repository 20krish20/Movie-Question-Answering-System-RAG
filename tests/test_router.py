import pytest

from movie_rag.pipelines.router import classify_query_type


@pytest.mark.parametrize("q", [
    "what is the average rating of action movies after 2010",
    "how many movies have rating greater than 8.5",
    "top 5 highest grossing movies directed by Steven Spielberg",
    "longest movies of all time",
    "movies released between 1990 and 1999",
    "which director has the most movies with rating above 8",
    "average runtime of horror movies released before 2000",
    "how many movies did martin scorsese direct",
    "average imdb score of pixar films",
])
def test_factual(q):
    assert classify_query_type(q) == "factual"


@pytest.mark.parametrize("q", [
    "movies about AI turning against humans",
    "a heist movie with a clever twist ending",
    "criminal masterminds rated R",  # "min" inside "criminal" must not trigger
    "the meaning of life comedies",  # "mean" inside "meaning"
    "a story set in New York after the war",
    "golden age hollywood romance",
])
def test_semantic(q):
    assert classify_query_type(q) == "semantic"
