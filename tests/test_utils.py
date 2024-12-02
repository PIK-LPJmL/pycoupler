"""Test the utils functions."""

from pycoupler.utils import search_country


def test_search_country():

    germany = search_country("germany")
    assert germany == "DEU"

    france = search_country("franz")
    assert france == "FRA"

    netherlands = search_country("nether")
    assert netherlands == "NLD"
