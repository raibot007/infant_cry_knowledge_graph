import os
import numpy as np
import soundfile as sf

# Mapping of subtype to bit depth
SUBTYPE_TO_BITS = {
    "PCM_U8": 8,
    "PCM_S8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "U8": 8,  # handles weird case you encountered
}

# Path to your dataset root folder
dataset_path = r"D:\TarunBali\infant_cry_knowledge_graph\baby_chilanto"

# Dictionary to hold class-wise stats
stats = {}

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue
    
    durations = []
    sample_rates = set()
    bitrates = set()
    channels_set = set()
    
    for file in os.listdir(class_path):
        if not file.lower().endswith((".wav", ".flac", ".mp3")):
            continue
        file_path = os.path.join(class_path, file)
        
        try:
            # Use soundfile to get detailed info
            with sf.SoundFile(file_path) as f:
                sr = f.samplerate
                channels = f.channels
                subtype = f.subtype  # e.g., PCM_16, U8
                
                # Determine bit depth safely
                bit_depth = SUBTYPE_TO_BITS.get(subtype, 16)  # default 16 if unknown
                
                duration = len(f) / sr
                bitrate = sr * bit_depth * channels  # in bps
                
                durations.append(duration)
                sample_rates.add(sr)
                bitrates.add(bitrate)
                channels_set.add(channels)
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if durations:
        stats[class_name] = {
            "num_files": len(durations),
            "sample_rates_Hz": list(sample_rates),
            "channels": list(channels_set),
            "bitrate_bps": list(bitrates),
            "min_duration_sec": float(np.min(durations)),
            "max_duration_sec": float(np.max(durations)),
            "mean_duration_sec": float(np.mean(durations)),
            "median_duration_sec": float(np.median(durations)),
        }

# Print summary
for cls, details in stats.items():
    print(f"\nClass: {cls}")
    for k, v in details.items():
        print(f"  {k}: {v}")
