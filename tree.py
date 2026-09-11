"""
Decision Tree vs Random Forest - Simple Fruit Classifier
Classifies fruit as Apple or Orange based on Weight and Size.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Data -> [Weight(g), Size(cm)]
X = [
    [150, 7], [170, 7.5], [140, 6.5], [160, 7.2], [155, 7.1],
    [145, 6.9], [165, 7.4], [135, 6.6], [158, 7.0], [148, 6.8],   # Apples
    [130, 6.8], [180, 8], [190, 8.2], [120, 6], [200, 8.5],
    [175, 7.9], [185, 8.1], [125, 6.2], [195, 8.4], [128, 6.3]    # Oranges
]
y = (["Apple"] * 10) + (["Orange"] * 10)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Step 4: Random Forest Model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Step 5: Predict a New Fruit
new_fruit = [[165, 7.3]]
print("\nNew Fruit [Weight=165g, Size=7.3cm] Prediction:")
print("Decision Tree says:", dt_model.predict(new_fruit)[0])
print("Random Forest says:", rf_model.predict(new_fruit)[0])