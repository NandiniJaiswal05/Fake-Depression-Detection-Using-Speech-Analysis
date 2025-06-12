import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from model import X_train
# Import preprocessing functions from your script (ensure the script is saved in the same directory)
from data_preprocessing import (
    extract_audio_features, 
    extract_formant_features, 
    load_formant_csv, 
    load_covarep_csv, 
    extract_text_features
)

### Load Pretrained Model ###
def load_model(model_path):
    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to scale features and reduce dimensionality to 54
def scale_and_reduce_features(combined_features):
    scaler = StandardScaler()

    # Print shape before scaling
    print("Shape of combined features before scaling:", combined_features.shape)
    
    # Scale features
    scaled_features = scaler.fit_transform(combined_features)
    
    # Print shape after scaling
    print("Shape of scaled features:", scaled_features.shape)

    # Get the number of samples and features
    n_samples, n_features = scaled_features.shape
    
    # Check if we have more than one sample for PCA
    if n_samples < 2:
        print("Insufficient samples for PCA. Returning scaled features.")
        return scaled_features

    # Set n_components dynamically based on the number of features
    n_components = min(54, n_samples, n_features)
    print(f"Reducing features to {n_components} components.")
    
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(scaled_features)
    
    return reduced_features

def scale_features_individually(audio_features, formant_features_wav, formant_features_csv, covarep_features, text_features):
    # Combine all features into one vector
    combined_features = np.concatenate((audio_features, 
                                         formant_features_wav, 
                                         formant_features_csv, 
                                         covarep_features, 
                                         text_features))
    
    print("Combined features shape before reduction:", combined_features.shape)
    
    # Reshape combined_features to a 2D array where the single sample has all features
    combined_features = combined_features.reshape(1, -1)  # Shape becomes (1, n_features)

    print("Shape of combined features for PCA:", combined_features.shape)
    
    # Scale and reduce features
    reduced_features = scale_and_reduce_features(combined_features)
    
    return reduced_features

### Test a Single Participant ###
# Function to scale features and reduce dimensionality using loaded scaler and PCA
def scale_and_reduce_features(combined_features, scaler, pca):
    # Print shape before scaling
    print("Shape of combined features before scaling:", combined_features.shape)
    
    # Scale features
    scaled_features = scaler.transform(combined_features)
    
    # Print shape after scaling
    print("Shape of scaled features:", scaled_features.shape)

    # Reduce features using PCA
    reduced_features = pca.transform(scaled_features)
    
    return reduced_features

# Update your test_single_participant function
def test_single_participant(wav_file, formant_csv, covarep_csv, transcript_file, model, scaler, pca):
    # Preprocess all files to extract features
    audio_features = extract_audio_features(wav_file)
    formant_features_wav = extract_formant_features(wav_file)
    formant_features_csv = load_formant_csv(formant_csv)
    covarep_features = load_covarep_csv(covarep_csv)
    text_features = extract_text_features(transcript_file)
    
    # Check if any feature extraction failed
    if (audio_features is None or formant_features_wav is None or 
        formant_features_csv is None or covarep_features is None or 
        text_features is None):
        print("Error: Missing data, cannot predict.")
        return None
    
    # Combine all features into one vector
    combined_features = np.concatenate((audio_features, 
                                         formant_features_wav, 
                                         formant_features_csv, 
                                         covarep_features, 
                                         text_features))
    
    # Reshape combined_features to a 2D array where the single sample has all features
    combined_features = combined_features.reshape(1, -1)  # Shape becomes (1, n_features)

    print("Shape of combined features for scaling and PCA:", combined_features.shape)
    
    # Scale and reduce features using loaded scaler and PCA
    reduced_features = scale_and_reduce_features(combined_features, scaler, pca)
    
    # Check if combined features can be reshaped correctly
    if reduced_features.ndim == 1:
        reduced_features = reduced_features.reshape(1, -1)  # Reshape to (1, n_features)
    
    # Make the prediction
    prediction = model.predict(reduced_features)
    
    # Get the prediction result (true/false/no depression)
    if prediction == 2:
        result = "True Depression"
    elif prediction == 1:
        result = "False Depression"
    else:
        result = "No Depression"
    
    print(f"Prediction for the participant: {result}")
    return result

if __name__ == "__main__":
    # Set the paths for files and the saved model
    model_path = r"C:\Users\nandi\Desktop\nandini\Program\Mini project\best_ensemble_model.pkl"  # Update with the actual path to your saved model
    scaler_path = r"C:\Users\nandi\Desktop\nandini\Program\Mini project\scaler.pkl"  # Path to saved scaler
    pca_path = r"C:\Users\nandi\Desktop\nandini\Program\Mini project\pca.pkl"  # Path to saved PCA
    wav_file = r"C:\Users\nandi\Downloads\310_P\310_AUDIO.wav"
    formant_csv = r"C:\Users\nandi\Downloads\310_P\310_FORMANT.csv"
    covarep_csv = r"C:\Users\nandi\Downloads\310_P\310_COVAREP.csv"
    transcript_file = r"C:\Users\nandi\Downloads\310_P\310_TRANSCRIPT.csv"
    # Load the pretrained model
    model = load_model(model_path)
    
    # Load the scaler and PCA
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    
    if model is not None:
        # Test on a single participant's data
        test_single_participant(wav_file, formant_csv, covarep_csv, transcript_file, model, scaler, pca)

