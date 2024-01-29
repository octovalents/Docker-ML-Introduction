import pandas as pd

from joblib import dump
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

def train():
  # Load and read data
  train_dir = "./train.csv"
  train_csv = pd.read_csv(train_dir)
  train_csv.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

  # drop null values
  train_csv.dropna(inplace=True)

  y_train = train_csv['Survived']
  x_train = train_csv.drop(['Survived'], axis=1)

  print('-- Shape of the training data')
  print(f'x_train: {x_train.shape}')
  print(f'y_train: {y_train.shape}')

  # normalize the data
  x_train = preprocessing.normalize(x_train, norm='l2')

  # Models training
  #-- Decision Tree
  clf_dt = DecisionTreeClassifier(random_state=0)
  clf_dt.fit(x_train, y_train)

  #-- save model
  dump(clf_dt, 'inference_dt.joblib')

if __name__ == '__main__':
  train()