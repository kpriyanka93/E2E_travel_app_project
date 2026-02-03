# Install important libraries
#%pip install pandas scikit-learn joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("C:/E2E_TRAVEL_project/dataset/flights.csv")

df.head()

#EDA
df.info()

# Check the Missing Values
df.isnull().sum()

"""The given dataset contains no missing values. Hence, no imputation is required."""

#Statistical Summary
df.describe()

# Visualisation of Target Variable Distribution (Price)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
sns.histplot(df["price"], bins=30, kde=True)
plt.title("Flight Price Distribution")
plt.show()

"""The prices are right-skewed, indicating presence of some expensive flights."""

# Visualisation of Flight Type distribution
plt.figure(figsize=(5,3))
sns.countplot(x="flightType", data=df)
plt.title("Flight Type Distribution")
plt.show()

#Price vs Flight Type
plt.figure(figsize=(6,4))
sns.boxplot(x="flightType", y="price", data=df)
plt.title("Price vs Flight Type")
plt.show()

"""Business/First class flights show higher median prices compared to economy."""

# Correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Select features
features = ["from","to","flightType","time","distance","agency"]
target = "price"
X = df[features]
y = df[target]

# Encode categorical columns
encoder = LabelEncoder()
for col in ["from","to","flightType","agency"]:
    X[col] = encoder.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred)
print("RMSE:", rmse)

# Save model
joblib.dump(model, "flight_model.pkl")
joblib.dump(encoder, "encoder.pkl")