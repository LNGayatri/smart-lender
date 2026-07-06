 from sklearn.tree import DecisionTreeClassifier
def decisionTree(X_train, X_test, y_train):
 model = DecisionTreeClassifier(random_state=42)
 model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     return y_pred
from sklearn.tree import DecisionTreeClassifier

def decisionTree(X_train, X_test, y_train):

    # Create the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test data
    y_pred = model.predict(X_test)

    # Return predictions
    return y_pred