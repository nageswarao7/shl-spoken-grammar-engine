import os
import sys
import subprocess

# Ensure openai-whisper is installed
try:
    import whisper
except ImportError:
    print("Installing openai-whisper...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"])
    import whisper

import glob
import time
import re
import numpy as np
import pandas as pd
import torch

print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper base.en model on {device}...")
model = whisper.load_model("base.en", device=device)

# Locate dataset folder
DATA_DIR = "/kaggle/input/shl-hiring-assessment-2026/Dataset_Final"
if not os.path.exists(DATA_DIR):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "Dataset_Final" in dirs:
            DATA_DIR = os.path.join(root, "Dataset_Final")
            break

print("Using DATA_DIR:", DATA_DIR)

FILLER_WORDS = {"um", "uh", "er", "ah", "like", "so", "well", "actually", "basically"}

def compute_nlp_metrics(text):
    if not text or len(text.strip()) == 0:
        return {
            "transcription": "",
            "word_count": 0,
            "char_count": 0,
            "avg_word_len": 0.0,
            "unique_words": 0,
            "ttr": 0.0,
            "filler_count": 0,
            "filler_ratio": 0.0,
            "sentence_count": 0,
            "avg_sentence_len": 0.0
        }
    
    words = re.findall(r"\b[A-Za-z']+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    word_count = len(words)
    char_count = sum(len(w) for w in words)
    avg_word_len = char_count / word_count if word_count > 0 else 0.0
    unique_words = len(set(words))
    ttr = unique_words / word_count if word_count > 0 else 0.0
    
    fillers = sum(1 for w in words if w in FILLER_WORDS)
    filler_ratio = fillers / word_count if word_count > 0 else 0.0
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
    
    return {
        "transcription": text,
        "word_count": word_count,
        "char_count": char_count,
        "avg_word_len": avg_word_len,
        "unique_words": unique_words,
        "ttr": ttr,
        "filler_count": fillers,
        "filler_ratio": filler_ratio,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len
    }

def transcribe_folder(folder_name):
    audio_dir = os.path.join(DATA_DIR, folder_name)
    files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"\n--- Transcribing {folder_name}: {len(files)} files ---")
    records = []
    t0 = time.time()
    
    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        try:
            res = model.transcribe(fpath, fp16=torch.cuda.is_available(), language="en")
            text = res.get("text", "").strip()
            segments = res.get("segments", [])
            avg_logprob = float(np.mean([s["avg_logprob"] for s in segments])) if segments else 0.0
            no_speech_prob = float(np.mean([s["no_speech_prob"] for s in segments])) if segments else 0.0
            
            row = {"filename": fname}
            row.update(compute_nlp_metrics(text))
            row["asr_avg_logprob"] = avg_logprob
            row["asr_no_speech_prob"] = no_speech_prob
            row["num_segments"] = len(segments)
            records.append(row)
        except Exception as e:
            row = {"filename": fname}
            row.update(compute_nlp_metrics(""))
            row["asr_avg_logprob"] = -99.0
            row["asr_no_speech_prob"] = 1.0
            row["num_segments"] = 0
            row["error"] = str(e)
            records.append(row)
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
            print(f"[{folder_name}] {idx+1}/{len(files)} completed in {time.time()-t0:.1f}s")
            
    df = pd.DataFrame(records)
    out_file = f"{folder_name}_transcriptions.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file} with shape {df.shape}")
    return df

if __name__ == "__main__":
    train_df = transcribe_folder("train")
    test_df = transcribe_folder("test")
    print("Done! Both train and test transcriptions and text features are saved.")
