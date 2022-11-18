from iris_inference_service.iris import Iris

import numpy as np

import pytest


@pytest.fixture
def iris():
    iris = Iris(model_name="Iris", model_file="./models/iris-model.pkl")
    return iris


def test_iris_attributes(iris):
    assert iris.model_name == "Iris"
    assert iris.model_file == "./models/iris-model.pkl"


def test_iris_load_model(iris):
    assert iris.loaded is False
    iris.load()
    assert iris.loaded is True


def test_iris_load_model_fail():
    iris = Iris(model_file="fake-file.pkl")
    assert iris.loaded is False
    with pytest.raises(Exception):
        iris.load()


def test_predict_response_wrong_number_of_x(iris):
    X = [[7.2, 3.6, 6.1]]
    columns = ["sepal length", "sepal width", "petal length", "petal width"]

    with pytest.raises(Exception):
        iris.predict(X=X, features_names=columns)


def test_model_response_types(iris):
    X = [[7.2, 3.6, 6.1, 2.5]]
    columns = ["sepal length", "sepal width", "petal length", "petal width"]

    result = iris.predict(X=X, features_names=columns)
    assert type(result) == np.ndarray
    assert len(result) == 1


# def test_request(iris, mocker):
#     X = [[7.2, 3.6, 6.1, 2.5]]
#     columns = ["sepal length", "sepal width", "petal length", "petal width"]

#     with mocker.patch.object(Iris, "model.predict", new=mock_predict):
#         result = iris.predict(X=X, features_names=columns)
#         assert result == [1]


def test_health_check(iris, mocker):
    mocker.patch.object(Iris, "predict", return_value=np.array([1]))
    result = iris.health_status()
    assert result == [1]


def test_health_check_wrong_length(iris, mocker):
    mocker.patch.object(Iris, "predict", return_value=np.array([1, 2]))
    with pytest.raises(AssertionError):
        iris.health_status()


def test_health_check_wrong_type(iris, mocker):
    mocker.patch.object(Iris, "predict", return_value=[1])
    with pytest.raises(AssertionError):
        iris.health_status()
