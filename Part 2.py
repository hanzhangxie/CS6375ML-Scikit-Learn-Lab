import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

cancer = pd.read_csv('breast-cancer-data.csv')

# pre-processing the dataset
cancer = cancer.dropna()
cancer = cancer.drop_duplicates()

# read in the dataset using Pandas into a dataframe
df = pd.DataFrame(cancer, columns=['Sample code number', 'Clump Thickness', 
	              'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
	              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
	              'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

# randomly split the data into train and test parts
X = cancer.iloc[:, 0:10]
y = cancer.iloc[:, 10:]

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()

# fit only to the training data
scaler.fit(X_train)

# apply the transformations to the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create a NN model
mlp = MLPClassifier(hidden_layer_sizes=(3,3,3))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))