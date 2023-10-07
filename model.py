import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the training data
df = pd.read_csv("train.csv")

# Split the data into features and target
X = df.drop("target", axis=1)
y = df["target"]

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Save the model to a file
model.save("model.pkl")
