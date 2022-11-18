# mlops-demo-iris-inference-service

This repository contains the assets for the Iris machine learning model and its component functions

## /notebooks

_seldon-request.ipynb_ contains functions for interacting with the ML service. This notebook acts as a client service. They send inputs to the service via http requests and validate the responses which are returned, comparing them against expected values.

_train_iris.ipynb_ uses the RandomForestClassifier to train on the Iris data set. The dataset is split into a training and testing set for identifying the species of plant based on Iris data.

## /notebooks/tests

Contains pytest and testbook framework tests for testing the notebooks. 

_test_seldon_request.py_ performs unit tests on the _seldon-request.ipynb_ notebook. The testbook framework allows for the loading of a notebook into a Python script. The pytest framework is then used to supply test cases and assert expected responses.
