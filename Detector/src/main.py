import torch
import argparse
from sklearn.preprocessing import LabelEncoder

from predict import predict, accents
from get_audio import extract_audio
from audio_maven import AudioMaven

def load_model(model_path = "../models/AudioMaven.pth"):
    torch.serialization.add_safe_globals({"AudioMaven": AudioMaven})
    model = torch.load(model_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), weights_only=False)
    return model

def main():
    parser = argparse.ArgumentParser(description="Predict spoken accent in a YouTube video.")
    parser.add_argument('--url', type=str, required=True, help='YouTube video URL')
    args = parser.parse_args()

    
    label_encoder = LabelEncoder()
    label_encoder.fit(accents)


    print(f"Downloading and extracting audio from: {args.url}")
    audio_path = extract_audio(args.url)

    print("Loading model...")
    model = load_model()

    print("Predicting accent...")
    predicted_accent, confidence = predict(audio_path, model, label_encoder)

    print(f"Predicted Accent: {predicted_accent}")
    print(f"Confidence Score: {confidence:.2%}")


if __name__ == "__main__":
    main()