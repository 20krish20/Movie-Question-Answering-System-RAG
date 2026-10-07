import pandas as pd
import pytest


@pytest.fixture
def movies():
    return pd.DataFrame({
        "Title": ["Inception", "Interstellar", "The Dark Knight", "Toy Story", "Get Out"],
        "Year": [2010, 2014, 2008, 1995, 2017],
        "Genres": ["Action, Sci-Fi", "Adventure, Drama, Sci-Fi", "Action, Crime, Drama", "Animation, Comedy", "Horror, Thriller"],
        "Certificate": ["PG-13", "PG-13", "PG-13", "G", "R"],
        "Runtime": [148.0, 169.0, 152.0, 81.0, 104.0],
        "Rating": [8.8, 8.7, 9.0, 8.3, 7.8],
        "Metascore": [74.0, 74.0, 84.0, 95.0, 85.0],
        "Votes": [2400000, 2000000, 2700000, 1000000, 650000],
        "Gross(Million)": [292.6, 188.0, 534.9, 191.8, 176.0],
        "Director": ["Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "John Lasseter", "Jordan Peele"],
        "Stars": ["Leonardo DiCaprio", "Matthew McConaughey", "Christian Bale", "Tom Hanks", "Daniel Kaluuya"],
        "Summary": ["Dream heist.", "Space travel.", "Batman vs Joker.", None, "Creepy in-laws."],
        "text": ["t1", "t2", "t3", "t4", "t5"],
    })
