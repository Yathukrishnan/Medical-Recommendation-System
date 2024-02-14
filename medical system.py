import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/contentdrive/MyDrive/int247-machine-learning-project-2020-kem031-sudhanshu-master/Dataset/training.csv')

# Data preprocessing
X = data.drop(columns=['prognosis'])  # Features (symptoms)
y = data['prognosis']  # Target variable (prognosis)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Function to get user input and make prediction
def predict_disease(user_symptoms):
    prediction = classifier.predict([user_symptoms])[0]
    return prediction

# Display symptoms and get user input
print("Please enter the symptoms (0 for absent, 1 for present):")
for i, symptom in enumerate(X.columns):
    print(f"{i + 1}. {symptom}")

user_symptoms = [0] * len(X.columns)  # Initialize user symptoms list with zeros

# Allow user to enter symptoms
while True:
    symptom_no = int(input("Enter the symptom number: "))
    user_symptoms[symptom_no - 1] = 1  # Mark symptom as active
    continue_entry = input("Do you want to continue entering symptoms? (y/n): ")
    if continue_entry.lower() != 'y':
        break

# Predict disease based on user symptoms
prediction = predict_disease(user_symptoms)
print("Predicted disease:", prediction)

# Calculate and print accuracy
accuracy = classifier.score(X_test, y_test)
print("Model Accuracy:", accuracy*100)

# Plot accuracy graph
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.title('Model Accuracy')
plt.xlabel('Metric')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


