import pandas as pd

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [50, 55, 60, 65, 70, 75, 85, 90],
    "marks": [20, 25, 40, 45, 60, 65, 80, 90]
}

df = pd.DataFrame(data)

print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print("Missing values:")
print(df.isnull().sum())

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Check the dataset again
print("Dataset after cleaning:")
print(df)

# Step 3: Separate features and target

X = df[["study_hours", "attendance"]]
y = df["marks"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Training features:")
print(X_train)

print("\nTesting features:")
print(X_test)

print("\nTraining targets:")
print(y_train)

print("\nTesting targets:")
print(y_test)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))