import config
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import json
import joblib


def preprocess_data(df, dependent_feature):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(["text", "Embeddings"], axis=1)
    encoder = LabelEncoder()
    df[dependent_feature] = encoder.fit_transform(df[dependent_feature])
    return df

def split_data(X, y, split_ratio=0.2, random_statee=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_statee, stratify=y)
    return X_train, X_test, y_train, y_test

def handle_imbalance(X, y):
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    models = {
        "tree": DecisionTreeClassifier(),
        "rf" : RandomForestClassifier(),
        "lgbm" : LGBMClassifier(),
        "cat" : CatBoostClassifier(),
        "xgb" : XGBClassifier(),
        "logreg" : LogisticRegression(max_iter=300),
        "svc" : SVC(kernel="rbf", gamma=0.5, C=0.1)
    }

    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train, y_train)
        print(f"{model} Training Completed")
        key = list(models.keys())
        print(key[i])
        model_out_path = config.MODEL_OUT_PATH+key[i]+"_imb"
        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        print(f"{model} Prediction done")
        y_pred.to_csv(model_out_path+".csv", index=False)

        metric_dict = {
            f"accuracy_{model}":float(accuracy_score(y_test, y_pred)),
            f"precision_score_{model}":float(precision_score(y_test, y_pred, average="weighted")),
            f"recall_score_{model}":float(recall_score(y_test, y_pred, average="weighted")),
            f"f1_score_{model}":float(f1_score(y_test, y_pred, average="weighted")),
            f"confusion_matrix_{model}":confusion_matrix(y_test, y_pred).tolist()
        }
        print(f"{model} Metric Calculation done")
        out_file = open(model_out_path+".json", "w")
        json.dump(metric_dict, out_file, indent = 6)
        print(f"{model} Metric Saving Done")
        out_file.close()

        joblib.dump(model, model_out_path+".pkl")
        print(f"{model} Saving Done")





    