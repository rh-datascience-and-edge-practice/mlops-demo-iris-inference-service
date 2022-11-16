import pytest
from testbook import testbook
import os

# from seldonrequest import get_iris_species
@pytest.fixture(scope="module")
def tb():
    with testbook("seldon-request.ipynb", execute=True) as tb:
        yield tb


dir_path = os.path.dirname(os.path.realpath(__file__))

# @testbook('/Users/atef/Documents/Bench/mlops-demo-iris/mlops-demo-iris-inference-service-main/notebooks/seldon-request.ipynb', execute=True)
@testbook(dir_path + "/seldon-request.ipynb", execute=True)
def test_get_iris_species(tb):
    get_iris_species = tb.ref("get_iris_species")
    assert get_iris_species(0) == "Iris-setosa"
    assert get_iris_species(1) == "Iris-versicolor"
    assert get_iris_species(2) == "Iris-virginica"


# # Test using function call.
# def test_get_iris_species():
#     assert get_iris_species(0) == "Iris-setosa"
#     assert get_iris_species(1) == "Iris-versicolor"
#     assert get_iris_species(2) == "Iris-virginica"


# # Test using function call.
# def test_get_iris_classification(seldontestbook):
#     get_iris_classification = seldontestbook.ref("get_iris_classification")
#     x =  '{ "name":"John", "age":30, "city":"New York"}'

#     assert get_iris_classification([1, 2, 3]) == [2, 4, 6]


# def test_double_array_inject(tb):
#     double_array = tb.ref("double_array")

#     tb.inject("""
#         data = [1, 2, 3]
#     """)
#     data = tb.ref("data")

#     assert double_array(data) == [2, 4, 6]
