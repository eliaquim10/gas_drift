from mimetypes import init
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# auc scores
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.decomposition import PCA



class MyPCA(BaseEstimator, TransformerMixin):
    def __init__(self, length):
        super()
        self.length = length
        self.value = []
        self.arg = []
        self.vector = []

        # self.arg = arg
    def fit(self, X, y):
        z = X - X.mean()
        cov = np.cov(z.T)
        self.value, self.vector = np.linalg.eig(cov)
        self.arg = np.argsort(self.value)[::-1]
        self.vector = self.vector[:, self.arg]

        return self
    def transform(self, X):
        z = X - X.mean()
        W = z.dot(self.vector)
        return W[:,:self.length]

        
    def fit_transform(self, X, y=None, **fit_params):
        # print(X)
        self.fit(X, y)
        filtering = self.transform(X)

        return filtering
    def get_value(self):
        return self.value

# import pandas as pd
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def classifier(model_name, model: BaseEstimator, X_train, y_train, X_test, y_test, under_sample: EditedNearestNeighbours = None, params_pca=None):
  print("="*30, model_name, "="*30)
  if under_sample:
    X_train, y_train = under_sample.fit_resample(X_train, y_train)
  if params_pca: 
    pipeline = Pipeline([
      ("std", StandardScaler()),
      ("pca", PCA(**params_pca)),
      ("model", model),
    ])
  else:
    pipeline = Pipeline([
      ("std", StandardScaler()),
      ("model", model),
    ])
    
  time_fit = datetime.now()
  
  pipeline.fit(X_train, y_train)
  time_fit = datetime.now() - time_fit

  time_predict = datetime.now()
  y_predict = pipeline.predict(X_test)
  y_pred_prob = pipeline.predict_proba(X_test)
  time_predict = datetime.now() - time_predict
  

  acc = accuracy_score(y_test, y_predict)
  roc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
  f1 = f1_score(y_test, y_predict, average=None).mean()
  cm = confusion_matrix(y_test, y_predict)
  
  print(model.best_params_) # TODO verificar hiper paramentros
  metricas = [roc, acc, f1, cm, time_fit.total_seconds(), time_predict.total_seconds()]
  
  # print(classification_report(y_test, y_predict))
  return model.best_params_, metricas.copy()

# dentro do kfold ou depois
def evaluate_models(models, paramentros: dict, X, y, cv: StratifiedKFold, under_sample=None, params_pca: dict=None):
  # np.mean()
  models_summary = []
  
  # interação do modelo 
  for model, paramentro in zip(models, paramentros):
    summary_roc = []
    summary_acc = []
    summary_f1 = []
    summary_confusion = []
    summary_time_fit = []
    summary_time_predict = []
    best_params = []
    for train_index, test_index in cv.split(X, y):
      # separa os dados de treino e teste da interação do k-fold
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      print(len(X_train), len(X_test))

      print("~"*10, model.get_params().keys(),"~"*10)
      gs = GridSearchCV(model, paramentro, verbose=0) 

      best_param, [roc, acc, f1, confusion, time_fit, time_predict] = classifier("{}".format(model), gs, X_train, y_train, X_test, y_test, under_sample, params_pca)
      summary_roc.append(roc)
      summary_acc.append(acc)
      summary_f1.append(f1)
      summary_confusion.append(confusion)
      summary_time_fit.append(time_fit)
      summary_time_predict.append(time_predict)
      
      best_params.append(best_param)

    models_summary.append([
      np.array(summary_roc).mean(),
      np.array(summary_acc).mean(),
      np.array(summary_f1).mean(),
      np.array(summary_time_fit).mean(),
      np.array(summary_time_predict).mean(),
      np.array(summary_confusion).mean(axis=0),
      best_params,
      str(model),
    ])
  return models_summary

