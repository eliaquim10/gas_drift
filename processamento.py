from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# auc scores
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from datetime import datetime


# import pandas as pd
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def classifier(model_name, model: BaseEstimator, X_train, y_train, X_test, y_test):
  print("="*30, model_name, "="*30)
  time = datetime.now()
  
  model.fit(X_train, y_train)
  time = datetime.now() - time

  y_predict = model.predict(X_test)
  y_pred_prob = model.predict_proba(X_test)

  acc = accuracy_score(y_test, y_predict)
  roc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
  f1 = f1_score(y_test, y_predict, average=None).mean()
  cm = confusion_matrix(y_test, y_predict)
  
  print(model.best_params_) # TODO verificar hiper paramentros
  metricas = [roc, acc, f1, cm, time.total_seconds()]
  
  # print(classification_report(y_test, y_predict))
  return model.best_params_, metricas.copy()

# dentro do kfold ou depois
def evaluate_models(models, paramentros, X, y, cv):
  # np.mean()
  models_summary = []
  
  # interação do modelo 
  for model, paramentro in zip(models, paramentros):
    try:
      summary_roc = []
      summary_acc = []
      summary_f1 = []
      summary_confusion = []
      summary_time = []
      best_params = []
      for train_index, test_index in cv.split(X, y):
        # separa os dados de treino e teste da interação do k-fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(len(X_train), len(X_test))

        print("~"*10, model.get_params().keys(),"~"*10)
        gs = GridSearchCV(model, paramentro, verbose=0) 

        best_param, [roc, acc, f1, confusion, time] = classifier("{}".format(model), gs, X_train, y_train, X_test, y_test)
        summary_roc.append(roc)
        summary_acc.append(acc)
        summary_f1.append(f1)
        summary_confusion.append(confusion)
        summary_time.append(time)
        
        best_params.append(best_param)

      models_summary.append([
        np.array(summary_roc).mean(),
        np.array(summary_acc).mean(),
        np.array(summary_f1).mean(),
        np.array(summary_time).mean(),
        np.array(summary_confusion).mean(axis=0),
        best_params,
        str(model),
      ])
    except:
      pass
    # resultados = pd.DataFrame(gs.cv_results_)
    # print(resultados.sort_values("rank_test_score", ascending=True).iloc[:5])
  # print("models_summary", np.mean(np.array(models_summary)))
  # print("models_summary `{}".format(models_summary))
  return models_summary


##############

def evaluate_models_to_svm(models, paramentros, X, y, cv):
  # np.mean()
  models_summary = []
  
  # interação do modelo 
  for model, paramentro in zip(models, paramentros):
    summary_roc = []
    summary_acc = []
    summary_f1 = []
    summary_confusion = []
    summary_time = []
    best_params = []
    for train_index, test_index in cv.split(X, y):
      # separa os dados de treino e teste da interação do k-fold
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      print(len(X_train), len(X_test))

      print("~"*10, model.get_params().keys(),"~"*10)
      gs = GridSearchCV(model, paramentro, verbose=0) 

      best_param, [roc, acc, f1, confusion, time] = classifier("{}".format(model), gs, X_train, y_train, X_test, y_test)
      summary_roc.append(roc)
      summary_acc.append(acc)
      summary_f1.append(f1)
      summary_confusion.append(confusion)
      summary_time.append(time)
      
      best_params.append(best_param)

    models_summary.append([
      np.array(summary_roc).mean(),
      np.array(summary_acc).mean(),
      np.array(summary_f1).mean(),
      np.array(summary_time).mean(),
      np.array(summary_confusion).mean(axis=0),
      best_params,
      str(model),
    ])
    # resultados = pd.DataFrame(gs.cv_results_)
    # print(resultados.sort_values("rank_test_score", ascending=True).iloc[:5])
  # print("models_summary", np.mean(np.array(models_summary)))
  # print("models_summary `{}".format(models_summary))
  return models_summary


