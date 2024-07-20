import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('ai4i2020.csv')

# Display first few rows
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Drop rows with missing values
df.dropna(inplace=True)

# Basic statistics for outlier detection
print(df.describe())

# Check column names
print("Columns in the dataset:", df.columns.tolist())

# Define features and target label
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
target = 'Machine failure'

# Verify if specified features exist in the DataFrame
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print("Missing features in the DataFrame:", missing_features)
else:
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']),
            ('cat', OneHotEncoder(), ['Type'])
        ])

    # Pipeline with RandomForest
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=model_pipeline.classes_)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_pipeline.classes_).plot()
    plt.show()

    # Feature importance visualization
    feature_importances = model_pipeline.named_steps['classifier'].feature_importances_
    feature_names = preprocessor.transformers_[0][2] + list(preprocessor.transformers_[1][1].get_feature_names_out(['Type']))
    importance_series = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)

    # Plot feature importances
    importance_series.plot(kind='bar')
    plt.title('Feature Importances')
    plt.show()
