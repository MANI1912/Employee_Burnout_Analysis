import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the employee dataset
df = pd.read_csv("G:\\Employee_Burnout_Analysis\\employeedataset.csv")

# Binning 'Burn Rate' into discrete categories (e.g., Low, Moderate, High)
def categorize_burn_rate(burn_rate):
    if burn_rate < 0.7:
        return 'Low'
    elif 0.7 <= burn_rate < 0.9:
        return 'Moderate'
    else:
        return 'High'

df['Burn Rate Category'] = df['Burn Rate'].apply(categorize_burn_rate)

# Data preprocessing
# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Company Type'] = label_encoder.fit_transform(df['Company Type'])
df['WFH Setup Available'] = label_encoder.fit_transform(df['WFH Setup Available'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])
df['Burn Rate Category'] = label_encoder.fit_transform(df['Burn Rate Category'])

# Separate features and target variable
X = df.drop(['Burn Rate', 'Burn Rate Category', 'Employee ID', 'Date of Joining'], axis=1)
y = df['Burn Rate Category']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Classification using Logistic Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)

# Classification using Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)

# Classification using Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

# Linear Regression for Resource Allocation and Burn Rate
regressor = LinearRegression()
regressor.fit(df[['Resource Allocation']], df['Burn Rate'])

# Evaluate the models
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))
print(confusion_matrix(y_test, lr_predictions))

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))
print(confusion_matrix(y_test, rf_predictions))

print("\nSupport Vector Machine (SVM) Results:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))
print(confusion_matrix(y_test, svm_predictions))

# Plot the graphs
plt.figure(figsize=(20, 12))

# Plot Histograms for 'Mental Fatigue Score' and 'Burn Rate'
plt.subplot(2, 3, 1)
sns.histplot(df['Mental Fatigue Score'], kde=True, color='blue')
plt.title('Histogram - Mental Fatigue Score')

plt.subplot(2, 3, 2)
sns.histplot(df['Burn Rate'], kde=True, color='green')
plt.title('Histogram - Burn Rate')

# Plot PCA scatter plot
plt.subplot(2, 3, 3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.8)
plt.title('PCA Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot Pair Plot for numerical attributes
numerical_attributes = ['Resource Allocation', 'Mental Fatigue Score', 'Burn Rate']
numerical_df = df[numerical_attributes + ['Burn Rate Category']]
sns.pairplot(numerical_df, hue='Burn Rate Category', palette='coolwarm')
plt.suptitle('Pair Plot for Numerical Attributes', y=1.02)

# Plot Correlation Heatmap for numerical attributes
plt.subplot(2, 3, 6)
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for Numerical Attributes')

# Plot bar plots for 'Gender' vs. other attributes
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
sns.barplot(x='Gender', y='Resource Allocation', data=df, palette='coolwarm')
plt.title('Mean Resource Allocation by Gender')

plt.subplot(2, 3, 2)
sns.barplot(x='Gender', y='Mental Fatigue Score', data=df, palette='coolwarm')
plt.title('Mean Mental Fatigue Score by Gender')

plt.subplot(2, 3, 3)
sns.barplot(x='Gender', y='Burn Rate', data=df, palette='coolwarm')
plt.title('Mean Burn Rate by Gender')

# Plot box plots for 'Gender' vs. other attributes
plt.subplot(2, 3, 4)
sns.boxplot(x='Gender', y='Resource Allocation', data=df, palette='coolwarm')
plt.title('Resource Allocation by Gender')

plt.subplot(2, 3, 5)
sns.boxplot(x='Gender', y='Mental Fatigue Score', data=df, palette='coolwarm')
plt.title('Mental Fatigue Score by Gender')

plt.subplot(2, 3, 6)
sns.boxplot(x='Gender', y='Burn Rate', data=df, palette='coolwarm')
plt.title('Burn Rate by Gender')

# Plot Linear Regression for Resource Allocation and Burn Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Resource Allocation', y='Burn Rate', data=df, color='purple')
sns.lineplot(x=df['Resource Allocation'], y=regressor.predict(df[['Resource Allocation']]), color='orange')
plt.title('Linear Regression: Resource Allocation vs. Burn Rate')
plt.xlabel('Resource Allocation')
plt.ylabel('Burn Rate')

plt.tight_layout()
plt.show()
