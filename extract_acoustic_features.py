import os
import glob
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor

DATA_DIR = r"e:\Assessment\shl_kaggle\data\Dataset_Final"
OUTPUT_DIR = r"e:\Assessment\shl_kaggle\data\features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_file_features(fpath):
    fname = os.path.basename(fpath)
    try:
        y, sr = sf.read(fpath)
        if len(y) == 0:
            return {"filename": fname, "error": 1}
        
        # Audio length & amplitude
        duration = len(y) / sr
        max_amp = float(np.max(np.abs(y)))
        
        # Frame settings: 25ms window, 10ms hop
        frame_len = int(sr * 0.025)
        hop_len = int(sr * 0.010)
        
        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_max = float(np.max(rms))
        
        # Silence & pause analysis
        # Threshold: 1% of peak RMS or absolute 0.001
        silence_thresh = max(0.01 * rms_max, 0.001)
        silent_frames = (rms < silence_thresh)
        silence_ratio = float(np.mean(silent_frames))
        
        # Group contiguous silent frames into pauses
        diff = np.diff(silent_frames.astype(int))
        pause_starts = np.where(diff == 1)[0]
        pause_ends = np.where(diff == -1)[0]
        num_pauses = len(pause_starts)
        
        # Spectral features
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_len)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_len)[0]
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len)
        
        feat = {
            "filename": fname,
            "duration": duration,
            "max_amp": max_amp,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "rms_max": rms_max,
            "silence_ratio": silence_ratio,
            "num_pauses": num_pauses,
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "spec_cent_mean": float(np.mean(spec_cent)),
            "spec_cent_std": float(np.std(spec_cent)),
            "spec_bw_mean": float(np.mean(spec_bw)),
            "spec_bw_std": float(np.std(spec_bw)),
            "spec_roll_mean": float(np.mean(spec_roll)),
            "spec_roll_std": float(np.std(spec_roll)),
            "is_silent_audio": 1 if (max_amp < 0.005 or rms_mean < 0.001 or silence_ratio > 0.95) else 0,
            "error": 0
        }
        
        for i in range(13):
            feat[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            feat[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
            
        return feat
    except Exception as e:
        return {"filename": fname, "error": 1, "msg": str(e)}

def process_dataset(name, file_list):
    print(f"Extracting acoustic features for {name} ({len(file_list)} files)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(extract_file_features, file_list))
    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_acoustic_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {name} features to {out_path} in {time.time()-t0:.1f}s. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.wav")))
    test_files = sorted(glob.glob(os.path.join(DATA_DIR, "test", "*.wav")))
    
    train_feat = process_dataset("train", train_files)
    test_feat = process_dataset("test", test_files)
    print("Done!")
