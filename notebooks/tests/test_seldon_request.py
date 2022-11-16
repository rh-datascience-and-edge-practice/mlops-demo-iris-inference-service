
import pytest
from testbook import testbook
import os 

# from seldonrequest import get_iris_species
@pytest.fixture(scope='module')
def tb():
    with testbook('seldon-request.ipynb', execute=True) as tb:
        yield tb

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

@testbook(parent_path + '/seldon-request.ipynb', execute=True)
def test_get_iris_species(tb):
    get_iris_species = tb.ref("get_iris_species")
    assert get_iris_species(0) == "Iris-setosa"
    assert get_iris_species(1) == "Iris-versicolor"
    assert get_iris_species(2) == "Iris-virginica"