import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve

# Load the dataset
file_path = 'Mall_Customers.csv'  # Update the file path as needed
data = pd.read_csv(file_path)

# Display column names to verify
print("Column Names in Dataset:")
print(data.columns)

# Strip any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Verify column names after cleaning
print("\nCleaned Column Names:")
print(data.columns)

# Display basic information
print("\nFirst 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
data.info()

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Basic statistics
print("\nDataset Statistics:")
print(data.describe())

# Selecting relevant columns for clustering
try:
    features = data[['Age', 'Annual_Income_(k$)', 'Spending_Score']]
except KeyError as e:
    print(f"Error: {e}")
    print("Ensure that the column names in the dataset match exactly: 'Age', 'Annual_Income_(k$)', 'Spending_Score'")
    exit()

# Elbow method to find the optimal number of clusters
inertia = []
range_clusters = range(1, 11)
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Choose optimal number of clusters (e.g., 5 based on the elbow method)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(features)

# Visualizing the clusters
plt.figure(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, optimal_clusters))  # Dynamically generate distinct colors
for i in range(optimal_clusters):
    plt.scatter(
        data.loc[data['Cluster'] == i, 'Annual_Income_(k$)'],
        data.loc[data['Cluster'] == i, 'Spending_Score'],
        s=100,
        color=colors[i],
        label=f'Cluster {i}'
    )
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()

# Logistic Regression: Ensure 'Genre' column exists
if 'Genre' in data.columns:
    # Encode 'Genre' as binary (Male = 0, Female = 1)
    data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})
else:
    print("Warning: 'Genre' column not found. Skipping Genre encoding.")

# Create binary target (mocked for demonstration)
# Replace this with true labels if available
target = data['Cluster'] % 2  # Example: Even clusters as 0, odd clusters as 1

# Features for Logistic Regression
X = data[['Age', 'Annual_Income_(k$)', 'Spending_Score']]
y = target

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Predictions
y_pred = logistic_model.predict(X_test)
y_pred_prob = logistic_model.predict_proba(X_test)[:, 1]

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid(True)
plt.show()
