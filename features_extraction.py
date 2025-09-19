import librosa
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call
import warnings
from pathlib import Path
import json
from tqdm import tqdm
import soundfile as sf

warnings.filterwarnings('ignore')

class ComprehensiveAudioFeatureExtractor:
    def __init__(self, sample_rate=16000, frame_length=2048, hop_length=512):
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mfcc = 13
        self.n_chroma = 12
        
        # For infant-specific analysis
        self.f0_min = 75   # Minimum F0 for infants
        self.f0_max = 800  # Maximum F0 for infants
        
    def load_audio(self, file_path):
        """Load and validate audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr)
            if len(audio) == 0:
                return None, f"Empty audio file: {file_path}"
            return audio, "Success"
        except Exception as e:
            return None, f"Error loading {file_path}: {str(e)}"
    
    def extract_time_domain_features(self, audio):
        """Extract time domain features (15 features)"""
        features = {}
        
        # 1. Amplitude-based features (5)
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                 hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length,
                                               hop_length=self.hop_length)[0]
        
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['amplitude_modulation_depth'] = np.std(rms) / (np.mean(rms) + 1e-10)
        
        # 2. Temporal structure features (5)
        # Autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # Find first significant peak after lag 0
        peaks, _ = find_peaks(autocorr[1:], height=0.1)
        features['autocorr_peak_value'] = np.max(autocorr[peaks + 1]) if len(peaks) > 0 else 0
        features['autocorr_peak_lag'] = (peaks[0] + 1) / self.sr if len(peaks) > 0 else 0
        
        # Temporal centroid
        time_frames = np.arange(len(rms)) / self.sr
        features['temporal_centroid'] = np.sum(time_frames * rms) / (np.sum(rms) + 1e-10)
        
        # Silence ratio
        silence_threshold = 0.01 * np.max(rms)
        features['silence_ratio'] = np.sum(rms < silence_threshold) / len(rms)
        
        # Burst density
        energy_peaks, _ = find_peaks(rms, height=0.5 * np.mean(rms), distance=10)
        features['burst_density'] = len(energy_peaks) / (len(audio) / self.sr)
        
        # 3. Statistical moments (5)
        features['amplitude_mean'] = np.mean(np.abs(audio))
        features['amplitude_std'] = np.std(audio)
        features['amplitude_skewness'] = stats.skew(audio)
        features['amplitude_kurtosis'] = stats.kurtosis(audio)
        features['dynamic_range'] = np.max(np.abs(audio)) - np.min(np.abs(audio))
        
        return features
    
    def extract_frequency_domain_features(self, audio):
        """Extract frequency domain features (25 features)"""
        features = {}
        
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # 4. Spectral shape features (8)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr,
                                                             hop_length=self.hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr,
                                                              hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr,
                                                          hop_length=self.hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio,
                                                            hop_length=self.hop_length)[0]
        
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        
        # Spectral flux
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        
        # Spectral slope and moments
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)
        spectral_slope = []
        spectral_skewness = []
        spectral_kurtosis = []
        
        for frame in magnitude.T:
            if np.sum(frame) > 0:
                # Spectral slope
                slope, _ = np.polyfit(freqs, frame, 1)
                spectral_slope.append(slope)
                
                # Spectral moments
                normalized_spectrum = frame / (np.sum(frame) + 1e-10)
                spectral_skewness.append(stats.skew(normalized_spectrum))
                spectral_kurtosis.append(stats.kurtosis(normalized_spectrum))
        
        features['spectral_slope_mean'] = np.mean(spectral_slope) if spectral_slope else 0
        features['spectral_skewness_mean'] = np.mean(spectral_skewness) if spectral_skewness else 0
        features['spectral_kurtosis_mean'] = np.mean(spectral_kurtosis) if spectral_kurtosis else 0
        
        # 5. MFCC features (13)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                                   hop_length=self.hop_length)
        for i in range(self.n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # 6. Harmonic features (4) - using librosa's harmonic analysis
        try:
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr,
                                                 fmin=self.f0_min, fmax=self.f0_max)
            
            # Extract fundamental frequency
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)
            
            if f0_values:
                features['f0_mean'] = np.mean(f0_values)
                features['f0_std'] = np.std(f0_values)
            else:
                features['f0_mean'] = 0
                features['f0_std'] = 0
            
            # Harmonicity estimation
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = np.sum(harmonic ** 2) / (np.sum(audio ** 2) + 1e-10)
            features['harmonicity'] = harmonic_ratio
            features['pitch_strength'] = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0
            
        except:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['harmonicity'] = 0
            features['pitch_strength'] = 0
        
        return features
    
    def extract_perceptual_features(self, audio):
        """Extract perceptual features (12 features)"""
        features = {}
        
        # 7. Psychoacoustic features (6)
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Loudness approximation (RMS in dB)
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        loudness_db = 20 * np.log10(rms + 1e-10)
        features['loudness_mean'] = np.mean(loudness_db)
        features['loudness_std'] = np.std(loudness_db)
        
        # Sharpness (high frequency emphasis)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)
        high_freq_mask = freqs > 2000  # Above 2kHz
        if np.any(high_freq_mask):
            sharpness = np.mean(magnitude[high_freq_mask, :], axis=0) / (np.mean(magnitude, axis=0) + 1e-10)
            features['sharpness_mean'] = np.mean(sharpness)
        else:
            features['sharpness_mean'] = 0
        
        # Roughness (amplitude modulation in 15-300 Hz range)
        # Simplified estimation using RMS fluctuation
        rms_diff = np.diff(rms)
        features['roughness'] = np.std(rms_diff)
        
        # Spectral irregularity
        spectral_irregularity = []
        for frame in magnitude.T:
            if len(frame) > 1:
                irregularity = np.sum(np.abs(np.diff(frame))) / (np.sum(frame) + 1e-10)
                spectral_irregularity.append(irregularity)
        features['spectral_irregularity'] = np.mean(spectral_irregularity) if spectral_irregularity else 0
        
        # Tonality coefficient (spectral flatness complement)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        features['tonality_coefficient'] = 1 - np.mean(spectral_flatness)
        
        # Fluctuation strength (low-frequency amplitude modulations)
        # Simplified as variation in short-term energy
        energy_frames = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
        features['fluctuation_strength'] = np.std(energy_frames) / (np.mean(energy_frames) + 1e-10)
        
        # 8. Advanced spectral features (6)
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr, hop_length=self.hop_length)
        for i in range(self.n_chroma):
            features[f'chroma_{i+1}'] = np.mean(chroma[i])
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr,
                                                            hop_length=self.hop_length)
        for i, band in enumerate(['sub_band', 'band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6']):
            if i < spectral_contrast.shape[0]:
                features[f'spectral_contrast_{band}'] = np.mean(spectral_contrast[i])
        
        return features
    
    def extract_infant_specific_features(self, audio):
        """Extract infant cry-specific features (8 features)"""
        features = {}
        
        try:
            # Use Parselmouth for precise voice analysis
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            
            # Pitch analysis
            pitch = sound.to_pitch(pitch_floor=self.f0_min, pitch_ceiling=self.f0_max)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 1:
                # Jitter (F0 variation)
                jitter = np.std(pitch_values) / (np.mean(pitch_values) + 1e-10)
                features['jitter'] = jitter
                
                # F0 range and contour
                features['f0_range'] = np.max(pitch_values) - np.min(pitch_values)
                features['f0_contour_slope'] = np.polyfit(np.arange(len(pitch_values)), pitch_values, 1)[0]
            else:
                features['jitter'] = 0
                features['f0_range'] = 0
                features['f0_contour_slope'] = 0
            
            # Shimmer (amplitude variation)
            intensity = sound.to_intensity()
            intensity_values = intensity.values.flatten()
            intensity_values = intensity_values[~np.isnan(intensity_values)]
            
            if len(intensity_values) > 1:
                shimmer = np.std(intensity_values) / (np.mean(intensity_values) + 1e-10)
                features['shimmer'] = shimmer
            else:
                features['shimmer'] = 0
            
            # Formant analysis
            try:
                formant = sound.to_formant_burg()
                f1_values = []
                f2_values = []
                f3_values = []
                
                for i in range(1, min(4, formant.get_number_of_formants() + 1)):
                    formant_track = [formant.get_value_at_time(i, t) 
                                   for t in np.linspace(0, sound.duration, 50)]
                    formant_track = [f for f in formant_track if not np.isnan(f) and f > 0]
                    
                    if formant_track:
                        if i == 1:
                            f1_values = formant_track
                        elif i == 2:
                            f2_values = formant_track
                        elif i == 3:
                            f3_values = formant_track
                
                features['f1_mean'] = np.mean(f1_values) if f1_values else 0
                features['f2_mean'] = np.mean(f2_values) if f2_values else 0
                features['f3_mean'] = np.mean(f3_values) if f3_values else 0
                
            except:
                features['f1_mean'] = 0
                features['f2_mean'] = 0
                features['f3_mean'] = 0
            
        except Exception as e:
            # Fallback to librosa-based analysis if Parselmouth fails
            print(f"Parselmouth analysis failed, using fallback: {e}")
            
            # Basic jitter estimation using pitch tracking
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr, 
                                                 fmin=self.f0_min, fmax=self.f0_max)
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)
            
            if len(f0_values) > 1:
                features['jitter'] = np.std(f0_values) / (np.mean(f0_values) + 1e-10)
                features['f0_range'] = np.max(f0_values) - np.min(f0_values)
                features['f0_contour_slope'] = np.polyfit(np.arange(len(f0_values)), f0_values, 1)[0]
            else:
                features['jitter'] = 0
                features['f0_range'] = 0
                features['f0_contour_slope'] = 0
            
            # Basic shimmer estimation
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features['shimmer'] = np.std(rms) / (np.mean(rms) + 1e-10)
            
            # Basic formant estimation (simplified)
            features['f1_mean'] = 0
            features['f2_mean'] = 0
            features['f3_mean'] = 0
        
        # Vocal effort estimation (energy concentration in higher frequencies)
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_length)
        
        # Energy in 1-4 kHz range (typical for cry effort)
        effort_freq_mask = (freqs >= 1000) & (freqs <= 4000)
        if np.any(effort_freq_mask):
            vocal_effort = np.mean(magnitude[effort_freq_mask, :]) / (np.mean(magnitude) + 1e-10)
            features['vocal_effort'] = vocal_effort
        else:
            features['vocal_effort'] = 0
        
        return features
    
    def extract_all_features(self, audio):
        """Extract all feature categories"""
        all_features = {}
        
        # Extract features from each category
        time_features = self.extract_time_domain_features(audio)
        freq_features = self.extract_frequency_domain_features(audio)
        perceptual_features = self.extract_perceptual_features(audio)
        infant_features = self.extract_infant_specific_features(audio)
        
        # Combine all features
        all_features.update(time_features)
        all_features.update(freq_features)
        all_features.update(perceptual_features)
        all_features.update(infant_features)
        
        return all_features
    
    def process_audio_file(self, file_path, file_id):
        """Process a single audio file and extract all features"""
        audio, status = self.load_audio(file_path)
        
        if audio is None:
            return {
                'file_id': file_id,
                'status': status,
                'feature_extraction_success': False
            }
        
        try:
            # Extract all features
            features = self.extract_all_features(audio)
            
            # Add metadata
            result = {
                'file_id': file_id,
                'status': 'Success',
                'feature_extraction_success': True,
                'audio_length_seconds': len(audio) / self.sr,
                **features
            }
            
            return result
            
        except Exception as e:
            return {
                'file_id': file_id,
                'status': f'Feature extraction failed: {str(e)}',
                'feature_extraction_success': False
            }
    
    def process_dataset_from_metadata(self, metadata_file_path, processed_audio_dir):
        """Process all files from preprocessing metadata"""
        # Load metadata
        if metadata_file_path.endswith('.csv'):
            metadata_df = pd.read_csv(metadata_file_path)
        else:
            metadata_df = pd.read_json(metadata_file_path)
        
        # Filter to only successfully preprocessed files
        successful_files = metadata_df[metadata_df['processed'] == True]
        
        print(f"Extracting features from {len(successful_files)} preprocessed audio files...")
        
        all_features = []
        failed_count = 0
        
        for idx, row in tqdm(successful_files.iterrows(), total=len(successful_files), 
                           desc="Extracting features"):
            file_path = Path(processed_audio_dir) / row['dataset'] / f"{row['file_id']}.wav"
            
            if file_path.exists():
                features = self.process_audio_file(file_path, row['file_id'])
                
                # Merge with original metadata
                merged_features = {**row.to_dict(), **features}
                all_features.append(merged_features)
                
                if not features['feature_extraction_success']:
                    failed_count += 1
            else:
                print(f"Warning: Processed audio file not found: {file_path}")
                failed_count += 1
        
        print(f"Feature extraction completed. {len(all_features)} files processed, {failed_count} failed.")
        
        return pd.DataFrame(all_features)
    
    def save_features(self, features_df, output_dir="features"):
        """Save extracted features"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        csv_path = output_path / "audio_features.csv"
        features_df.to_csv(csv_path, index=False)
        
        # Save as JSON for programmatic access
        json_path = output_path / "audio_features.json"
        features_df.to_json(json_path, orient='records', indent=2)
        
        # Save feature names list
        feature_columns = [col for col in features_df.columns 
                         if col not in ['file_id', 'dataset', 'class', 'original_filename',
                                      'original_path', 'processed_path', 'status',
                                      'feature_extraction_success']]
        
        feature_info = {
            'total_features': len(feature_columns),
            'feature_names': feature_columns,
            'feature_categories': {
                'time_domain': [f for f in feature_columns if any(td in f for td in 
                               ['rms', 'zcr', 'amplitude', 'autocorr', 'temporal', 'silence', 'burst', 'dynamic'])],
                'frequency_domain': [f for f in feature_columns if any(fd in f for fd in 
                                   ['spectral', 'mfcc', 'f0', 'harmonic', 'pitch'])],
                'perceptual': [f for f in feature_columns if any(p in f for p in 
                             ['loudness', 'sharpness', 'roughness', 'chroma', 'contrast', 'tonality', 'fluctuation'])],
                'infant_specific': [f for f in feature_columns if any(i in f for i in 
                                  ['jitter', 'shimmer', 'f1', 'f2', 'f3', 'vocal_effort'])]
            }
        }
        
        with open(output_path / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\nFeatures saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        print(f"  Feature info: {output_path / 'feature_info.json'}")
        
        return csv_path, json_path


# Usage example
def main():
    """Main feature extraction pipeline"""
    
    # Initialize feature extractor
    extractor = ComprehensiveAudioFeatureExtractor(sample_rate=16000)
    
    # Extract features from preprocessed data
    # Assuming you have the metadata from Phase 1
    metadata_file = r"D:\TarunBali\infant_cry_knowledge_graph\preprocessed_audio\preprocessing_metadata.csv"
    processed_audio_dir = r"D:\TarunBali\infant_cry_knowledge_graph\preprocessed_audio"
    
    if Path(metadata_file).exists():
        # Process all files
        features_df = extractor.process_dataset_from_metadata(
            metadata_file, processed_audio_dir
        )
        
        # Save features
        csv_path, json_path = extractor.save_features(features_df, output_dir=r"D:\TarunBali\infant_cry_knowledge_graph\features")
        
        # Print summary
        successful_features = features_df[features_df['feature_extraction_success'] == True]
        print(f"\n=== FEATURE EXTRACTION SUMMARY ===")
        print(f"Total files processed: {len(features_df)}")
        print(f"Successful extractions: {len(successful_features)}")
        print(f"Failed extractions: {len(features_df) - len(successful_features)}")
        print(f"Number of features per file: ~60")
        
        # Show feature categories
        feature_cols = [col for col in successful_features.columns 
                       if col not in ['file_id', 'dataset', 'class', 'original_filename',
                                    'original_path', 'processed_path', 'status',
                                    'feature_extraction_success']]
        print(f"Total feature dimensions: {len(feature_cols)}")
        
        print("\n=== Ready for Phase 3: Knowledge Graph Construction ===")
        
    else:
        print(f"Metadata file not found: {metadata_file}")
        print("Please run Phase 1 preprocessing first.")


if __name__ == "__main__":
    main()