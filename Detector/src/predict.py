import torch
import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Assuming the model and label_encoder are pre-loaded or passed to the function

accents = ['spanish', 'english', 'arabic', 'russian', 'dutch', 'bengali',
           'french', 'thai', 'mandarin', 'portuguese', 'mongolian',
           'italian', 'finnish', 'german', 'korean', 'greek', 'romanian'
            ]
mfcc = 13

def predict(audio_path, model, sample_rate=22050, n_mfcc=mfcc):

    label_encoder = LabelEncoder()
    label_encoder.fit(accents)

    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    mfcc_mean = np.mean(mfcc, axis=1)  # Mean across time axis (axis=1)
    mfcc_std = np.std(mfcc, axis=1)    # Standard deviation across time axis (axis=1)
    
    features = np.concatenate((mfcc_mean, mfcc_std))  # This gives a vector of length 26
    
    features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0).unsqueeze(2)  # Shape: (1, 26, 1)
    
    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        _, predicted_class = torch.max(output, dim=1)
    
    predicted_class_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]

    os.remove(audio_path)
    
    return predicted_class_label
