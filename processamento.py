from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import pandas as pd

def classifier(model_name, model: BaseEstimator, X_train, y_train, X_test, y_test):
  print("="*30, model_name, "="*30)
  model.fit(X_train, y_train)
  y_predict = model.predict(X_test)
  print(model.best_params_)
  metricas = [accuracy_score(y_test, y_predict), f1_score(y_test, y_predict, average=None).mean(), confusion_matrix(y_test, y_predict)]
  
  print(classification_report(y_test, y_predict))
  return metricas

# dentro do kfold ou depois
def evaluate_models(models, paramentros, X, y, cv):
  # np.mean()
  models_summary = []
  for model, paramentro in zip(models, paramentros):

    # interação do modelo 
    for train_index, test_index in cv.split(X, y):
      summary = []
      # separa os dados de treino e teste da interação do k-fold
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      gs = GridSearchCV(model, paramentro) 
      metricas = classifier("Regressão Logistica {}".format(model), gs, X_train, y_train, X_test, y_test)
      summary.append(metricas)
    models_summary.append(models_summary)
    # resultados = pd.DataFrame(gs.cv_results_)
    # print(resultados.sort_values("rank_test_score", ascending=True).iloc[:5])
  print("models_summary", np.mean(np.array(models_summary)))
  return models_summary

