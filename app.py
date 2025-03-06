import os
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import joblib
import tempfile

# Load the saved model and label encoder
model_path = 'model.pkl'
label_encoder_path = 'label_encoder.pkl'
if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
    raise FileNotFoundError("Model or label encoder not found. Please train the model first.")

clf = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

app = FastAPI(title="Infant Cry Analysis API")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr = librosa.feature.zero_crossing_rate(y)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=50)
    
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    zcr_mean = np.mean(zcr.T, axis=0)
    spec_contrast_mean = np.mean(spec_contrast.T, axis=0)
    
    return np.concatenate((mfccs_mean, chroma_mean, zcr_mean, spec_contrast_mean))

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    # Save the uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_file_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving temporary file.")
    
    try:
        # Extract features and make prediction
        feature = extract_features(temp_file_path)
        feature = feature.reshape(1, -1)
        prediction = clf.predict(feature)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing audio file.")
    finally:
        os.remove(temp_file_path)
    
    return JSONResponse(content={"predicted_reason": predicted_label})

if __name__ == "__main__":
    import uvicorn
    # For production, you might disable reload (or set it based on a config variable)
    uvicorn.run("app:app", host="0.0.0.0", port=3004, reload=False)
