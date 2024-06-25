import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Read and display the data
df = pd.read_csv("extracted_data.csv")
print(df.head())
print(df.dtypes)

# Load the preprocessed data
preprocessed_data = pd.read_csv("subset_preprocessed_data.csv")

# EDA: Count of Adverse Reactions
plt.figure(figsize=(8, 6))
sns.countplot(x='Adverse_Reaction', data=preprocessed_data, palette='pastel')
plt.title('Count of Adverse Reactions')
plt.xlabel('Adverse Reaction')
plt.ylabel('Count')
plt.show()

# EDA: Drug vs Adverse Reaction
plt.figure(figsize=(10, 6))
sns.countplot(y='Drug_Name', hue='Adverse_Reaction', data=preprocessed_data, palette='pastel', order=preprocessed_data['Drug_Name'].value_counts().index[:10])
plt.title('Drug vs Adverse Reaction')
plt.xlabel('Count')
plt.ylabel('Drug Name')
plt.legend(title='Adverse Reaction')
plt.show()

# EDA: Adverse Reaction vs Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Adverse_Reaction', hue='Sex', data=preprocessed_data, palette='pastel')
plt.title('Adverse Reaction vs Gender')
plt.xlabel('Adverse Reaction')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()

# EDA: Adverse Reaction vs Seriousness Death
plt.figure(figsize=(8, 6))
sns.countplot(x='Adverse_Reaction', hue='Seriousness_Death', data=preprocessed_data, palette='pastel')
plt.title('Adverse Reaction vs Seriousness Death')
plt.xlabel('Adverse Reaction')
plt.ylabel('Count')
plt.legend(title='Seriousness Death')
plt.show()

# EDA: Adverse Reaction vs Seriousness Hospitalization
plt.figure(figsize=(8, 6))
sns.countplot(x='Adverse_Reaction', hue='Seriousness_Hospitalization', data=preprocessed_data, palette='pastel')
plt.title('Adverse Reaction vs Seriousness Hospitalization')
plt.xlabel('Adverse Reaction')
plt.ylabel('Count')
plt.legend(title='Seriousness Hospitalization')
plt.show()

print("EDA analysis with adverse reactions included.")

# Preprocess data for Random Forest
subset_preprocessed_data = pd.read_csv("subset_preprocessed_data.csv")
subset_preprocessed_data.columns = subset_preprocessed_data.columns.str.strip()
X = subset_preprocessed_data.drop(columns=['Adverse_Reaction'])
y = subset_preprocessed_data['Adverse_Reaction']
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

rf_classifier = RandomForestClassifier()
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_classifier)
])

pipeline.fit(X, y)
feature_importances = rf_classifier.feature_importances_
one_hot_encoded_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(input_features=categorical_cols)
feature_names = list(one_hot_encoded_feature_names) + numerical_cols
sorted_indices = feature_importances.argsort()[::-1]
n_selected_features = 10
top_feature_indices = sorted_indices[:n_selected_features]
selected_features = [feature_names[i] for i in top_feature_indices]

print("Selected Features and Importances:")
for feature, importance in zip(selected_features, feature_importances[top_feature_indices]):
    print(f"Feature: {feature}, Importance: {importance}")

plt.figure(figsize=(10, 6))
plt.bar(range(len(selected_features)), feature_importances[top_feature_indices], align="center")
plt.xticks(range(len(selected_features)), selected_features, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

print("\nAnalysis of Feature Importances:")

# Load data for RNN model
data = pd.read_csv("patient_data.csv")
features = ['precipitation', 'temp_max', 'temp_min', 'wind']
target = 'weather'
X = data[features]
y = data[target]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_rnn = tf.keras.utils.to_categorical(y_train)
y_test_rnn = tf.keras.utils.to_categorical(y_test)
X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model_rnn = Sequential([
    LSTM(64, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model_rnn.fit(X_train_rnn, y_train_rnn, validation_split=0.2, epochs=26, batch_size=32, callbacks=[early_stopping])

final_loss = history.history['val_loss'][-1]
final_accuracy = history.history['val_accuracy'][-1]
print("Final Loss:", final_loss)
print("Final Accuracy:", final_accuracy)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Train and evaluate Naive Bayes classifier
subset_preprocessed_data = pd.read_csv("subset_preprocessed_data.csv")
subset_preprocessed_data.columns = subset_preprocessed_data.columns.str.strip()
X = subset_preprocessed_data.drop(columns=['Adverse_Reaction'])
y = subset_preprocessed_data['Adverse_Reaction']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
encoder = OneHotEncoder(sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)
X_processed = pd.concat([X_encoded, X_scaled], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

y_pred_prob = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("AUC-ROC Score:", roc_auc)
