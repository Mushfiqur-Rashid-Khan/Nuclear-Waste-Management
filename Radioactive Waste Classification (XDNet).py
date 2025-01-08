# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten

# Load the dataset
file_path = '/content/Dataset_RW.csv'  # Replace with the actual path in your Colab
df = pd.read_csv(file_path)

# Convert 'I_131_(Bq/m3)' to numeric and handle any non-numeric values
df['I_131_(Bq/m3)'] = pd.to_numeric(df['I_131_(Bq/m3)'], errors='coerce')

# Drop rows with missing or NaN values
df_cleaned = df.dropna()

# Prepare features (Longitude, Latitude, Cs_134, Cs_137, I_131) and target (Code)
X = df_cleaned[['Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)']].values
y = df_cleaned['Code'].values

# Standardize the feature data for better neural network performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input data for Conv1D and LSTM layers (adding an extra dimension)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build the Extended DNN model
model = Sequential()

# 1D Convolutional Layer for spatial feature learning
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))

# LSTM layer for temporal feature learning
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))

# GRU layer for sequential learning
model.add(GRU(32))
model.add(Dropout(0.3))

# Flatten and Dense layers for final classification
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation for numeric target

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Predictions on test set
y_pred = model.predict(X_test)

# Example of comparing predictions with true values
print("True values: ", y_test[:5])
print("Predicted values: ", y_pred[:5])

pip install catboost

# Import CatBoost and necessary libraries
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten
from sklearn.metrics import mean_absolute_error


# Load the dataset
file_path = '/content/Dataset_RW.csv'  # Replace with the actual path in your Colab
df = pd.read_csv(file_path)

# Convert 'I_131_(Bq/m3)' to numeric and handle any non-numeric values
df['I_131_(Bq/m3)'] = pd.to_numeric(df['I_131_(Bq/m3)'], errors='coerce')

# Drop rows with missing or NaN values
df_cleaned = df.dropna()


# Data Preparation as before
X = df_cleaned[['Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)']].values
y = df_cleaned['Code'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost Model
catboost_model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
catboost_model.fit(X_train, y_train)

# CatBoost predictions
catboost_preds = catboost_model.predict(X_test)

# Extended DNN Model
# Preprocessing for DNN (standardizing and reshaping for Conv1D/LSTM/GRU)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# DNN model as defined earlier
dnn_model = Sequential()
dnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
dnn_model.add(Dropout(0.3))
dnn_model.add(LSTM(64, return_sequences=True))
dnn_model.add(Dropout(0.3))
dnn_model.add(GRU(32))
dnn_model.add(Dropout(0.3))
dnn_model.add(Flatten())
dnn_model.add(Dense(128, activation='relu'))
dnn_model.add(Dropout(0.3))
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dense(1, activation='linear'))

# Compile the model
dnn_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the DNN model
dnn_model.fit(X_train_reshaped, y_train, epochs=5, batch_size=64, validation_data=(X_test_reshaped, y_test))

# DNN predictions
dnn_preds = dnn_model.predict(X_test_reshaped)

# Combine CatBoost and DNN predictions (simple average)
final_preds = (catboost_preds + dnn_preds.flatten()) / 2

# Evaluate the final predictions
final_mae = mean_absolute_error(y_test, final_preds)
print(f"Final MAE: {final_mae}")

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/RW_Dataset.csv')

# Inspect the data (you might need to clean or preprocess it)
print(data.head())

# Assuming the relevant features for clustering are columns 'Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)'
X = data[['Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)']].values

# Standardize the data (DBSCAN is sensitive to scales, so normalization is important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
# eps: maximum distance between two samples for them to be considered as in the same neighborhood
# min_samples: minimum number of samples in a neighborhood for a point to be considered a core point
dbscan = DBSCAN(eps=0.5, min_samples=5)  # You can tweak eps and min_samples to get around 6 clusters
clusters = dbscan.fit_predict(X_scaled)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Save the results to a CSV file
output_file = '/content/DBSCAN_clustering_output.csv'
data.to_csv(output_file, index=False)

print(f'Clustering results saved to {output_file}')

# Visualize the clusters
# Reduce dimensionality for visualization using PCA (to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting the clusters
plt.figure(figsize=(10, 7))
unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], label=f'Cluster {cluster}')

plt.title('DBSCAN Clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Print the resulting number of clusters and their distribution
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # -1 is noise in DBSCAN
print(f"Number of clusters (excluding noise): {n_clusters}")
print("Cluster distribution:")
print(pd.Series(clusters).value_counts())



pip install pyclustering

pip install tslearn

# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/RW_Dataset.csv')

# Assuming the relevant features for clustering are 'Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)'
X = data[['Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Save the results to a CSV file
output_file = '/content/KMeans_clustering_output.csv'
data.to_csv(output_file, index=False)

print(f'Clustering results saved to {output_file}')

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.show()

# Print cluster sizes
print(data['Cluster'].value_counts())

"""**XDNet**"""

# Import necessary libraries
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd


# Load the dataset
data = pd.read_csv('/content/Dataset_RW.csv')

# Inspect the data (you might need to clean or preprocess it)
print(data.head())



# Assume df_cleaned is your preprocessed dataset
# Feature matrix (X) and target vector (y)
X = df_cleaned[['Longitude', 'Latitude', 'Cs_134_(Bq/m3)', 'Cs_137_(Bq/m3)', 'I_131_(Bq/m3)']].values
y = df_cleaned['Cluster'].values  # Assuming 'Code' is the 6-class label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)
catboost_model.fit(X_train, y_train)

# CatBoost predictions (get probabilities for classification)
catboost_probs = catboost_model.predict_proba(X_test)

# Preprocessing for DNN (standardizing and reshaping for Conv1D/LSTM/GRU)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# DNN model for multiclass classification
dnn_model = Sequential()
dnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
dnn_model.add(Dropout(0.3))
dnn_model.add(LSTM(64, return_sequences=True))
dnn_model.add(Dropout(0.3))
dnn_model.add(GRU(32))
dnn_model.add(Dropout(0.3))
dnn_model.add(Flatten())
dnn_model.add(Dense(128, activation='relu'))
dnn_model.add(Dropout(0.3))
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dense(6, activation='softmax'))  # Output layer with 6 classes

# Compile the DNN model
dnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the DNN model
dnn_model.fit(X_train_reshaped, y_train, epochs=20, batch_size=64, validation_data=(X_test_reshaped, y_test))

# DNN predictions (get probabilities for classification)
dnn_probs = dnn_model.predict(X_test_reshaped)

# Combine CatBoost and DNN predictions (simple average of probabilities)
final_probs = (catboost_probs + dnn_probs) / 2

# Convert probabilities to predicted classes
final_preds = np.argmax(final_probs, axis=1)

# Evaluate the combined model
accuracy = accuracy_score(y_test, final_preds)
print(f"Final Accuracy: {accuracy}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, final_preds)
print(f"Confusion Matrix:\n{conf_matrix}")

# Classification report (precision, recall, f1-score)
class_report = classification_report(y_test, final_preds, digits=4)
print(f"Classification Report:\n{class_report}")

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Assuming 'final_preds' contains the predicted labels and 'y_test' the true labels
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, final_preds)

# Plotting the colorful confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[f'Class {i}' for i in range(6)],
            yticklabels=[f'Class {i}' for i in range(6)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Display the classification report with accuracy, precision, recall, and F1-score
class_report = classification_report(y_test, final_preds, digits=4)
print(f"Classification Report:\n{class_report}")



# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Exempt Waste', 'Very Short-Lived Waste', 'Very Low Level Waste', 'Low Level Waste', ' Intermediate Level Waste', 'High Level Waste'],
            yticklabels=['Exempt Waste', 'Very Short-lived Waste', 'Very Low Level Waste', 'Low Level Waste', ' Intermediate Level Waste', 'High Level Waste'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Radioactive Waste Classification')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['EW', 'VSLW', 'VLLW', 'LLW', ' ILW', 'HLW'],
            yticklabels=['EW', 'VSLW', 'VLLW', 'LLW', ' ILW', 'HLW'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Radioactive Waste Classification')

# Extract True Positives, False Positives, False Negatives, and True Negatives for each class
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP
TN = np.sum(conf_matrix) - (FP + FN + TP)

# Calculate accuracy, precision, recall, and F1-score for each class
accuracy1 = np.sum(TP) / np.sum(conf_matrix)
accuracy= accuracy1*100
precision_weighted = np.sum(TP) / (np.sum(TP) + np.sum(FP))
recall_weighted = np.sum(TP) / (np.sum(TP) + np.sum(FN))
f1_score_weighted = 2 * (precision_weighted * recall_weighted) / (precision_weighted + recall_weighted)


# Print results
print(f"Accuracy: {accuracy:.3f} %")
print(f"Overall Precision: {precision_weighted:.5f}")
print(f"Overall Recall: {recall_weighted:.5f}")
print(f"Overall F1-Score: {f1_score_weighted:.5f}")

