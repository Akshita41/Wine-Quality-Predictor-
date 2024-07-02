import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# loading the dataset to pandas dataframe
wine_dataset = pd.read_csv('extracted_files/winequalityN.csv')

# Number of rows and columns in the dataset
print(wine_dataset.shape)

# first five rows of the datasets
print(wine_dataset.head())

# checking for null values
print(wine_dataset.isnull().sum())

# filling the null values with the mean of the column
# Exclude the 'type' column from mean imputation as it's non-numeric
numeric_columns = wine_dataset.select_dtypes(include=np.number).columns
wine_dataset[numeric_columns] = wine_dataset[numeric_columns].apply(lambda col: col.fillna(col.mean()), axis=0)

# description of the data set
print(wine_dataset.describe())

# number of values for each quality
plot1 = plt.figure(figsize=(5, 5))
sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.show()

# volatile acidity vs quality graph
plot = plt.figure(figsize=(6, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
plt.show()

# citric acid vs quality graph
plot2 = plt.figure(figsize=(6, 5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
plt.show()

# Select only numeric columns for correlation calculation
numeric_wine_dataset = wine_dataset.select_dtypes(include=np.number)
corelation = numeric_wine_dataset.corr()  # Calculate correlation on numeric data only

# constructing a heatmap to understand the co-relation between the columns
plt.figure(figsize=(10, 10))
sns.heatmap(corelation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# separation of data
x = wine_dataset.drop(['quality', 'type'], axis=1)
print(x)
y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
print(y)

# splitting data into test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(y.shape, y_train.shape, y_test.shape)

# training the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('The accuracy is : ', test_data_accuracy)

# building a predictive system
custom_input = (6.2, 0.66, 0.48, 1.2, 0.029, 29, 75, 0.9892, 3.33, 0.39, 12.8)
input_data = np.asarray(custom_input)
input_data_reshape = input_data.reshape(1, -1)
prediction = model.predict(input_data_reshape)
if prediction[0] == 1:
    print('The wine quality is good')
else:
    print('The wine quality is bad')
