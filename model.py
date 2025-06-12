import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Load Processed Data
with open(r"processed_data.pkl", 'rb') as f:
    processed_data, participant_ids = pickle.load(f)

# Split the processed data into features (X) and labels (y)
X = np.array([data[0] for data in processed_data])  # Features
y = np.array([data[1] for data in processed_data])  # PHQ-8 Scores

# Define classes based on PHQ-8 scores
y_classes = np.where(y <= 4, 0, np.where((y >= 5) & (y <= 14), 1, 2))

# Check for NaN and Inf in your dataset
mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
X = X[mask]
y_classes = y_classes[mask]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_classes, test_size=0.2, random_state=42, stratify=y_classes)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Impute missing values with mean for the training set
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Impute missing values for the test set using the same imputer
X_test = imputer.transform(X_test)

# Standardize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Ensemble Models: Random Forest and Gradient Boosting
rf_clf = RandomForestClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)

# Voting classifier to combine different models
voting_clf = VotingClassifier(estimators=[
    ('random_forest', rf_clf),
    ('gradient_boosting', gb_clf)
], voting='soft')

# Pipeline to automate scaling and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('voting_clf', voting_clf)     # Ensemble model
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'voting_clf__random_forest__n_estimators': [100, 200, 500],
    'voting_clf__gradient_boosting__n_estimators': [100, 200,350],
    'voting_clf__random_forest__max_depth': [None, 150, 200, 200],
    'voting_clf__gradient_boosting__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2, scoring='accuracy')

# Train the Model
print("Training the ensemble model with GridSearchCV...")
grid_search.fit(X_train_pca, y_train)

# Best model after GridSearchCV
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Save the Best Model
with open('best_ensemble_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best model saved successfully!")

# Evaluate the Model
print("Evaluating the model on the test set...")
y_pred = best_model.predict(X_test_pca)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

