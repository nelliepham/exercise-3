import string
import os
from PIL import Image
import numpy as np
import librosa

# ---------- TEXT ----------
def load_text_file(filepath):
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.lower()
            if line:
                lines.append(line)
    return lines

def tokenize(text):
    return text.split()

# ---------- IMAGE ----------
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images

def preprocess_image(img, desired_size=(224, 224)):
    img_resized = img.resize(desired_size)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr

# ---------- AUDIO ----------
def load_audio_file(file_path, sr=16000):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def extract_mfcc(audio, sr=16000, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# ---------- MAIN ----------
def main():
    # Text
    print("\n--- TEXT ---")
    text_data = load_text_file("dataset.txt")
    print(f"Loaded {len(text_data)} lines.")
    tokens = [tokenize(line) for line in text_data]
    print("Example tokens:", tokens[0])

    # Image
    print("\n--- IMAGES ---")
    imgs = load_images_from_folder("images")
    print(f"Loaded {len(imgs)} images.")
    if imgs:
        processed_img = preprocess_image(imgs[0])
        print("Processed image shape:", processed_img.shape)

    # Audio
    print("\n--- AUDIO ---")
    audio_data, sr = load_audio_file("example.wav")
    print(f"Loaded audio with {len(audio_data)} samples at {sr} Hz.")
    mfcc = extract_mfcc(audio_data, sr)
    print("MFCC shape:", mfcc.shape)

if __name__ == "__main__":
    main()
