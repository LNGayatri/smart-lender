 # Import Libraries
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==========================================
# Step 1: Create a Sample Dataset
# ==========================================

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[
    "Feature1",
    "Feature2",
    "Feature3",
    "Feature4",
    "Feature5"
])

df["Target"] = y

# ==========================================
# Step 2: Separate Features and Target
# ==========================================

X = df.drop("Target", axis=1)
y = df["Target"]

# ==========================================
# Step 3: Split the Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Step 4: Create Random Forest Function
# ==========================================

def randomForest(X_train, X_test, y_train):

    # Create the Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test data
    y_pred = model.predict(X_test)

    return y_pred

# ==========================================
# Step 5: Call the Function
# ==========================================

y_pred = randomForest(X_train, X_test, y_train)

# ==========================================
# Step 6: Evaluate the Model
# ==========================================

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))