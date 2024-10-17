import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the training data
print("Loading training data...")
train_data = pd.read_csv('train.csv')
print("Data loaded successfully.")

# Print the first few rows of the dataset
print("Training data preview:")
print(train_data.head())

# Check the columns in the dataset
print("Columns in the training data:", train_data.columns)

# Update these to the actual feature names you want to use
X = train_data[['LotArea', 'OverallQual', 'YearBuilt']]  # Example columns
y = train_data['SalePrice']  # Adjust the target variable name as needed

# Split the data into training and testing sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
print("Making predictions...")
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Predictions made.")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Prepare predictions for submission
test_data = pd.read_csv('test.csv')
print("Loading test data...")
print("Test data preview:")
print(test_data.head())

# Update these to match the test data structure
X_submit = test_data[['LotArea', 'OverallQual', 'YearBuilt']]  # Example columns
submission_predictions = model.predict(X_submit)

# Create a submission DataFrame
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Adjust if necessary
    'Price': submission_predictions
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
