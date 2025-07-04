# diabetes_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


df = pd.read_csv(r'E:\diabetes\diabetes.csv')
print("Dataset Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
sns.pairplot(df, hue='Outcome')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n[Logistic Regression Results]")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n[Random Forest Results]")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\n[SVM Results]")
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))

print("\n--- Diabetes Prediction from User Input ---")
try:
    pregnancies = float(input("Enter number of Pregnancies: "))
    glucose = float(input("Enter Glucose level: "))
    bp = float(input("Enter Blood Pressure: "))
    skin = float(input("Enter Skin Thickness: "))
    insulin = float(input("Enter Insulin level: "))
    bmi = float(input("Enter BMI: "))
    dpf = float(input("Enter Diabetes Pedigree Function: "))
    age = float(input("Enter Age: "))

    user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)

    0
    prediction = rf_model.predict(user_input_scaled)

    if prediction[0] == 1:
        print("The model predicts: The person is likely to have diabetes.")
    else:
        print("The model predicts: The person is NOT likely to have diabetes.")

except ValueError:
    print("Invalid input. Please enter numeric values only.")
