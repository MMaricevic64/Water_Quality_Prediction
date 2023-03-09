import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler

dataset = "dataset_2.csv"

def ML_input_output(df):
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    return X,Y

#------------------DATASET------------------------------#

#Dataset input
df = pd.read_csv(dataset)

#------------------Customising dataset_1.csv-----------------------#
if (dataset == "dataset_1.csv"):
    #Remove unnecessary columns (first and last two)
    del df[df.columns[0]]
    df = df.iloc[: , :-2]
    
    #Dataset columns and their types
    print(df.columns)
    print(df.dtypes)
    
    #Example 1 - Replace all Nan with 0
    #df1 = df.replace(np.nan, 0)
    
    #Example 2 - Remove all rows with at least 1 Nan value
    #df2 = df.dropna().reset_index(drop=True)
#-----------------------------------------------------------------#

#Define X and Y - input and output
X, Y = ML_input_output(df) #df - dataset_2.csv, df1 - dataset_1.csv,example 1, df2 - dataset_1.csv,example2

#Divide dataset for training and validation (70:30)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, shuffle=True)

#Normalize data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#------------------MODELS------------------------------#
#Define models
model = MLPClassifier(hidden_layer_sizes=(500, 300,), alpha=0.0001, solver='adam', activation="relu", max_iter=1000, verbose=True)
#model = KNeighborsClassifier(n_neighbors=5)

#Training and validation
model.fit(X_train, Y_train)

#Confusion matrix
plot_confusion_matrix(model, X_test, Y_test, normalize='true', display_labels=['Not potable', 'Potable'], cmap=plt.cm.Blues)
