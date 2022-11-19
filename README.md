# mlops-demo-iris-inference-service

This repository contains the assets for serving Iris machine learning model and its component functions.

## Running locally

### Installing pipenv

This repository utilizes pipenv for package management.  To install pipenv on your local machine run the following command:

```
pip install pipenv
```

### Virtual Environment and Dependencies

To install the dependencies for running the application, run the following command:

```
pipenv install --dev
```

To activate the virtual environment run the following command:

```
pipenv shell
```

### Starting the Application

To run the application run the following command:

```
python app.py
```

## Folders

### /iris_inference_service

This folder containers the main python package containing the source code utilized for serving the machine learning model.

### /models

The models folder contains the machine learning model trained by the train_iris.ipynb or another repository.  Models should be saved in the .pkl format.

### /notebooks

_seldon-request.ipynb_ contains functions for interacting with the ML service. This notebook acts as a client service. They send inputs to the service via http requests and validate the responses which are returned, comparing them against expected values.

_train_iris.ipynb_ uses the RandomForestClassifier to train on the Iris data set. The dataset is split into a training and testing set for identifying the species of plant based on Iris data.  This notebook can be used to help train a local version of the model.

### /tests

Tests contains test functions that can be executed using the pytest framework.

```
pytest tests/
```
