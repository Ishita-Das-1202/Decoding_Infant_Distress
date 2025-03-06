import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Update this path to point to your dataset location on macOS.
dataset_path = os.path.normpath('donateacry_corpus')

# Define categories
categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr = librosa.feature.zero_crossing_rate(y)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=50)
    
    # Aggregate features
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    zcr_mean = np.mean(zcr.T, axis=0)
    spec_contrast_mean = np.mean(spec_contrast.T, axis=0)
    
    return np.concatenate((mfccs_mean, chroma_mean, zcr_mean, spec_contrast_mean))

# Prepare the dataset
def prepare_data():
    features = []
    labels = []
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(category)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return np.array(features), np.array(labels)

def main():
    # Load and preprocess data
    X, y = prepare_data()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Create a pipeline: scaling + classifier
    clf = make_pipeline(StandardScaler(), 
                        RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, class_weight='balanced'))
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred, target_names=categories))
    
    # Save the model and the label encoder for later use in FastAPI
    joblib.dump(clf, 'model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model and label encoder saved to disk.")

if __name__ == "__main__":
    main()
