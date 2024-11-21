import os
import librosa
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    # Load the audio file
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Generate the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert to log scale (dB)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

def mel_spectrogram_to_image_tensor(mel_spectrogram, resize=(224, 224)):
    # Convert mel spectrogram (2D array) to a PIL Image
    mel_image = Image.fromarray(np.uint8((mel_spectrogram + 80) * 255 / 80))  # Normalizing to [0, 255]
    mel_image = mel_image.convert("RGB")
    
    # Resize and transform the image to tensor
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(mel_image)
    
    return image_tensor

def process_folder_to_tensors(folder_path, save_to_folder=None, resize=(224, 224)):
    tensors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_path = os.path.join(folder_path, filename)
            
            # Convert audio to mel spectrogram
            mel_spectrogram = audio_to_mel_spectrogram(audio_path)
            
            # Convert mel spectrogram to image tensor
            image_tensor = mel_spectrogram_to_image_tensor(mel_spectrogram, resize=resize)
            tensors.append(image_tensor)
            
            if save_to_folder:
                # Save the tensor if needed
                save_path = os.path.join(save_to_folder, f"{os.path.splitext(filename)[0]}.pt")
                torch.save(image_tensor, save_path)
    
    return tensors

import subprocess
from private import get_yt_dlt_folder
def get_playlist_mp3(playlist_url):

    # Call the batch file with arguments
    subprocess.run([get_yt_dlt_folder(), playlist_url, 'a'])


def main():
    # Define paths
    audio_folder = "path_to_audio_folder"  # Folder containing audio files
    save_folder = "path_to_save_tensors"   # Folder to save tensors, set to None if saving isn't needed
    
    # Ensure the save folder exists
    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Process audio files and convert them to image tensors
    mel_tensors = process_folder_to_tensors(audio_folder, save_to_folder=save_folder)

    # Print out some details for verification
    print(f"Processed {len(mel_tensors)} audio files.")
    print("Example tensor shape:", mel_tensors[0].shape if mel_tensors else "No tensors generated.")

    # Optional: Display the first tensor's details for inspection
    if mel_tensors:
        print("First tensor:", mel_tensors[0])
        
if __name__ == "__main__":
    main()
