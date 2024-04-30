import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

    def drop_na(self):
        self.data.dropna(inplace = True)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createXGBoostModel(n_estimators=200, max_depth = 5, learning_rate = 0.1)
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5


    def data_exploration(self):
        fig, ax = plt.subplots(nrows = 2, ncols=3, figsize = (12,8))
        sns.boxplot(y = "Age", x = self.output_data, data = self.input_data, ax = ax[0,0])
        ax[0,0].set_title("churn Based on Age", fontsize = 15)
        sns.boxplot(y = "CreditScore", x = self.output_data, data = self.input_data, ax = ax[0,1])
        ax[0,1].set_title("churn Based on CreditScore", fontsize = 15)
        sns.boxplot(y = "Tenure", x = self.output_data, data = self.input_data, ax = ax[0,2])
        ax[0,2].set_title("churn Based on Tenure", fontsize = 15)
        sns.boxplot(y = "Balance", x = self.output_data, data = self.input_data, ax = ax[1,0])
        ax[1,0].set_title("churn Based on Balance", fontsize = 15)
        sns.boxplot(y = "EstimatedSalary", x = self.output_data, data = self.input_data, ax = ax[1,1])
        ax[1,1].set_title("churn Based on EstimatedSalary", fontsize = 15)
        plt.tight_layout()
        plt.show()


    def drop_feature(self, columns):
        self.input_data.drop(columns= columns, inplace = True)

    def log_transform(self, columns):
        for i in columns:
            self.x_train[i] = np.log(self.x_train[i] + 1e-8)
            self.x_test[i] = np.log(self.x_test[i] + 1e-8)
        
    def standard_scaler(self, columns):
        ss = StandardScaler()
        self.x_train[columns]= ss.fit_transform(self.x_train[columns])
        self.x_test[columns]= ss.fit_transform(self.x_test[columns])

    def one_hot_encoding(self, columns):
        ohe = OneHotEncoder()
        self.x_train= self.x_train.reset_index()
        self.x_test=self.x_test.reset_index() 
        for column in columns:
            train_temp = pd.DataFrame(ohe.fit_transform(self.x_train[[column]]).toarray(),columns=ohe.get_feature_names_out())
            test_temp = pd.DataFrame(ohe.transform(self.x_test[[column]]).toarray(),columns=ohe.get_feature_names_out())
            self.x_train = pd.concat([self.x_train.drop(columns=[column]), train_temp], axis = 1)
            self.x_test = pd.concat([self.x_test.drop(columns=[column]), test_temp], axis = 1)

    def remove_outliers(self, columns):
        data = pd.concat([self.x_train, self.y_train], axis = 1)
        for i in columns:
            Q1 = np.quantile(data[i], 0.25)
            Q3 = np.quantile(data[i], 0.75)

            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            data = data[(data[i] >= lower_bound) & (data[i] <= upper_bound)]
            
        label =  pd.DataFrame(self.y_train).columns[0]
        self.x_train = data.drop(columns = label)
        self.y_train = data[label]

    def random_oversampling(self):         
        ros = RandomOverSampler()
        self.x_train, self.y_train = ros.fit_resample(self.x_train, self.y_train)

    def take_top_names(self, topnamesamount):
        topnames = topnamesamount
        names_to_keep = self.x_train["Surname"].value_counts()[:topnames]
        self.x_train["Surname"] = self.x_train["Surname"].apply(lambda x: x if x in names_to_keep else 'Other')
        self.x_test["Surname"] = self.x_test["Surname"].apply(lambda x: x if x in names_to_keep else 'Other')

    def createRandomForestModel(self, n_estimators = 300, criterion = "gini"):
        self.model = RandomForestClassifier(n_estimators= n_estimators, criterion=criterion)

    def createXGBoostModel(self, n_estimators=200, max_depth = 5, learning_rate = 0.1):
        self.model = XGBClassifier(n_estimators=n_estimators, max_depth = max_depth, learning_rate = learning_rate)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))
            
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.model, file) 


file_path = 'data_A.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.drop_na()
data_handler.create_input_output('churn')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
# model_handler.data_exploration()
model_handler.drop_feature(["Unnamed: 0", "id", "CustomerId"])

model_handler.log_transform(["Age", "Balance"])
model_handler.remove_outliers(["CreditScore", "Age", "Balance", "EstimatedSalary"])
model_handler.standard_scaler(["CreditScore", 'Age', 'Balance', 'EstimatedSalary', "Tenure"])
model_handler.take_top_names(10)
model_handler.one_hot_encoding(["Geography", "Gender", "HasCrCard", "IsActiveMember", "Surname"])
model_handler.random_oversampling()

model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()
# model_handler.save_model_to_file('XGB_class.pkl') 