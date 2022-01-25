from functools import reduce
from mimetypes import init
from sklearn.ensemble import RandomForestClassifier
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

def drop(lista: list, coluna):
  return [item for item in lista if (item[0]==coluna)]

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

        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        filtering = self.transform(X)

        return filtering
    def get_value(self):
        return self.value

class ENNModify(BaseEstimator, TransformerMixin):
  def __init__(self, params):
      super()
      self.params = params

      # self.arg = arg
  def fit(self, X, y):
      self.enn = EditedNearestNeighbours(**self.params)

      return self
  def transform(self, X):
    return X
      
  def fit_transform(self, X, y=None):
      self.fit(X, y)
      X, y = self.enn.fit_resample(X, y)
      return X, y
  
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        self.feature = []
        self.gsrfr = None
        self.feature_importances = []
        self.param_grid = [{
            'n_estimators': list(range(3, 20, 3)), 
            'max_features': list(range(10, 120, 10))
            },]

    
    def fit(self, X, y=None):
        self.gsrfr = GridSearchCV(
            RandomForestClassifier(random_state=42),
            self.param_grid,
            cv=5,
            return_train_score=True
        )
        self.gsrfr.fit(X, y) 
        self.feature_importances = self.gsrfr.best_estimator_.feature_importances_
        self.max_features = self.gsrfr.best_estimator_.max_features
        print("self.max_features", self.max_features)
        return self
    
    def transform(self, X, y=None):
        best_features = np.argsort(self.feature_importances)
        best_features = best_features[::-1]
        best_features = best_features[:self.max_features]
        return X[:, best_features]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        
        return  self.transform(X, y)

class ModelModify(BaseEstimator, TransformerMixin):
  def __init__(self, model, params_pca = None, params_enn = None, reverse = False):
    super()
    self.model = model
    self.transformers = []
    if(params_pca):
      self.pca = PCA(**params_pca)
    if(params_enn):
      self.enn = ENNModify(params_enn)
    self.reverse = reverse
    
    # self.arg = arg
  def fit(self, X, y):
    if (self.reverse):
      if hasattr(self, "enn"):
        X, y = self.enn.fit_transform(X, y)
      if hasattr(self, "pca"):
        X = self.pca.fit_transform(X, y)
    else:
      if hasattr(self, "pca"):
        X = self.pca.fit_transform(X, y)
      if hasattr(self, "enn"):
        X, y = self.enn.fit_transform(X, y)
    self.model.fit(X, y)
    return self
  def transform(self, X):
    return X
      
  def fit_transform(self, X, y=None):
    return X

  def predict(self, X):
    if hasattr(self, "pca"):
      X = self.pca.transform(X)
    return self.model.predict(X)

  def predict_proba(self, X):
    if hasattr(self, "pca"):
      X = self.pca.transform(X)
    return self.model.predict_proba(X)
# import pandas as pd
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def classifier(model_name, model: BaseEstimator, X_train, y_train, X_test, y_test, params_enn = None, params_pca=None, reverse=False):
  print("="*30, model_name, "="*30)
  
  pipeline = Pipeline([
      ("std", StandardScaler()),
      # ("fs", FeatureSelector(X_train.columns)),
      ("model", ModelModify(model, params_pca, params_enn, reverse)),
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
  
  return model.best_params_, metricas.copy()

# dentro do kfold ou depois
def evaluate_models(models, paramentros: dict, X, y, cv: StratifiedKFold, params_enn=None, params_pca=None, reverse=False):
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

      print("~"*10, model.get_params().keys(),"~"*10)
      gs = GridSearchCV(model, paramentro, verbose=0) 

      best_param, [roc, acc, f1, confusion, time_fit, time_predict] = classifier("{}".format(model), gs, X_train, y_train, X_test, y_test, params_enn, params_pca, reverse)
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

