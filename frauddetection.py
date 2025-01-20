#create fraud detection 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
data = pd.read_csv("./dataset/creditcard.csv")

#remove warning
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Preprocessing
# Load the data
# Separate features and target variable
X = data.drop(['Class', 'Amount'], axis=1)  # Exclude 'Class' and 'Amount' from features
y = data['Class']

# Handle imbalanced data using undersampling
undersampler = RandomUnderSampler()
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Step 2: Model Training
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 

# Step 3: Model Evaluation
# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

