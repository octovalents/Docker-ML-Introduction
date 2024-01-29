FROM jupyter/scipy-notebook

RUN mkdir my-model
ENV MODEL_DIR=/my-model
ENV MODEL_NAME=inference_dt.joblib

RUN pip install joblib pandas scikit-learn

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py