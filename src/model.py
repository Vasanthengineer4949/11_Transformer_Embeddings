from utils import *

class Model():

    def __init__(self, df):
        self.df = df
        self.dependent_feature = "label"

    def run(self):

        preprocessed_df = preprocess_data(self.df, self.dependent_feature)
        print("Preprocessing Completed")

        X = preprocessed_df.drop(self.dependent_feature, axis=1)
        y = preprocessed_df[self.dependent_feature]
        
        X_train, X_test, y_train, y_test = split_data(X, y, split_ratio=0.25, random_statee=42)
        print("Splitting Data done")

        X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
        print("Imbalance Handle done")

        train_model(X_train, y_train, X_test, y_test)

