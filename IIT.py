import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Define paths
AUDIO_DIR = "Audio"
LABELS_DIR = "Labels"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load audio file paths and corresponding transcripts
def load_data(audio_dir, labels_dir):
    audio_files = sorted(os.listdir(audio_dir))
    transcripts = {}
    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            with open(os.path.join(labels_dir, file), "r", encoding="utf-8") as f:
                transcripts.update({line.split()[0]: " ".join(line.split()[1:]) for line in f})
    return [os.path.join(audio_dir, file) for file in audio_files], transcripts

# Custom dataset class for Kathbath dataset
class KathbathDataset(Dataset):
    def __init__(self, audio_files, transcripts, tokenizer, transform=None):
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        transcript = self.transcripts[os.path.splitext(os.path.basename(audio_file))[0]]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, transcript

# Preprocessing and feature extraction
def preprocess_audio(waveform):
    mel_spectrogram = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=128)
    return mel_spectrogram(waveform)

# Calculate WER
def calculate_wer(ground_truth, predicted):
    total_words = sum(len(gt.split()) for gt in ground_truth)
    total_errors = 0
    for gt, pred in zip(ground_truth, predicted):
        total_errors += wer(gt, pred)
    return total_errors / total_words

def wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    d = np.zeros((len(ref_words) + 1) * (len(hyp_words) + 1), dtype=np.uint32).reshape((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        for j in range(len(hyp_words) + 1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(ref_words)][len(hyp_words)]

def main():
    # Load data
    audio_files, transcripts = load_data(AUDIO_DIR, LABELS_DIR)

    # Initialize tokenizer and model
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

    # Dataset and Dataloader
    dataset = KathbathDataset(audio_files, transcripts, tokenizer, transform=preprocess_audio)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Transcribe
    predictions = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            logits = model(inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        predicted_transcription = tokenizer.batch_decode(pred_ids)[0]
        predictions.append(predicted_transcription)

    # Calculate WER
    ground_truth = list(transcripts.values())
    wer_score = calculate_wer(ground_truth, predictions)
    print("Word Error Rate (WER):", wer_score)

if __name__ == "__main__":
    main()
