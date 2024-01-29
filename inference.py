import os

import pandas as pd

from joblib import load
from sklearn import preprocessing

def inference():
  MODEL_DIR = os.environ['MODEL_DIR']
  MODEL_NAME = os.environ['MODEL_NAME']
  MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

  test_dir = "./test.csv"
  test_csv = pd.read_csv(test_dir)
  test_csv.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

  # drop null values
  test_csv.dropna(inplace=True)

  print('-- Shape of the testing data')
  print(f'test_csv: {test_csv.shape}')

  # normalize the data
  test_csv = preprocessing.normalize(test_csv, norm='l2')

  # Models training
  #-- Decision Tree
  clf_dt = load(MODEL_PATH)
  print("Decision Tree classification:")
  print(clf_dt.predict(test_csv[:10]))

if __name__ == '__main__':
  inference()