import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, target_sr=16000, output_dir="preprocessed_audio"):
        self.target_sr = target_sr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each dataset
        self.datasets = ['donateacry', 'cried', 'baby_chilanto']
        for dataset in self.datasets:
            (self.output_dir / dataset).mkdir(exist_ok=True)
        
        self.metadata = []
        
    def load_and_validate_audio(self, file_path):
        """Load audio file and perform basic validation"""
        try:
            # Load audio with original sample rate
            audio, original_sr = librosa.load(file_path, sr=None)
            
            if len(audio) == 0:
                return None, None, "Empty audio file"
            
            # Check for extreme clipping (>95% samples at maximum amplitude)
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            if clipping_ratio > 0.1:  # More than 10% clipped samples
                return audio, original_sr, f"High clipping detected: {clipping_ratio:.2%}"
            
            return audio, original_sr, "Valid"
            
        except Exception as e:
            return None, None, f"Load error: {str(e)}"
    
    def calculate_audio_quality_metrics(self, audio, sr):
        """Calculate quality metrics for metadata"""
        # Signal-to-noise ratio estimation
        frame_length = min(2048, len(audio))
        energy = librosa.feature.rms(y=audio, frame_length=frame_length)[0]
        snr_estimate = 20 * np.log10(np.max(energy) / (np.mean(energy) + 1e-10))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
        
        # Zero crossing rate (indicates noisiness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        
        return {
            'snr_estimate_db': round(snr_estimate, 2),
            'dynamic_range_db': round(dynamic_range, 2),
            'zero_crossing_rate': round(zcr, 4)
        }
    
    def preprocess_audio(self, audio, original_sr):
        """Apply minimal preprocessing"""
        # Step 1: Resample to target sample rate
        if original_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        
        # Step 2: Remove DC offset
        audio = audio - np.mean(audio)
        
        # Step 3: Gentle amplitude normalization (preserve relative loudness)
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            # Normalize to 80% of full scale to prevent clipping
            audio = audio * (0.8 / max_amplitude)
        
        # Step 4: Light noise gate (remove pure silence, not quiet sounds)
        # Only remove segments where RMS is extremely low (likely digital silence)
        rms_threshold = 0.001  # Very conservative threshold
        rms_values = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        if np.mean(rms_values) < rms_threshold:
            # File is mostly silence - flag but don't modify
            pass
        
        return audio
    
    def categorize_duration(self, duration):
        """Categorize file duration for metadata"""
        if duration < 2.0:
            return "short"
        elif duration < 10.0:
            return "medium"
        else:
            return "long"
    
    def extract_donateacry_metadata(self, filename):
        """Extract metadata from DonateACry filename format"""
        # Format: UUID-timestamp-version-gender-age-class_code
        # Example: 69BDA5D6-0276-4462-9BF7-951799563728-1436936185-1.1-m-26-bp
        try:
            parts = filename.replace('.wav', '').replace('.mp3', '').split('-')
            if len(parts) >= 6:
                uuid = '-'.join(parts[:5])
                timestamp = parts[5] if parts[5].isdigit() else None
                version = parts[6] if len(parts) > 6 else None
                gender = parts[7] if len(parts) > 7 else None
                age = parts[8] if len(parts) > 8 and parts[8].isdigit() else None
                class_code = parts[9] if len(parts) > 9 else None
                
                # Map gender codes
                gender_full = {'m': 'male', 'f': 'female'}.get(gender, gender)
                
                # Map class codes
                class_mapping = {
                    'bp': 'belly_pain', 'bu': 'burping', 'dc': 'discomfort', 
                    'hu': 'hungry', 'ti': 'tired'
                }
                class_from_code = class_mapping.get(class_code, class_code)
                
                return {
                    'uuid': uuid,
                    'timestamp': int(timestamp) if timestamp else None,
                    'version': version,
                    'gender': gender_full,
                    'age_months': int(age) if age else None,
                    'class_code': class_code,
                    'class_from_filename': class_from_code
                }
        except Exception as e:
            print(f"Error parsing DonateACry filename {filename}: {e}")
        
        return {}
    
    def extract_cried_metadata(self, filename):
        """Extract metadata from CRIED filename format"""
        # Format: f{participant_id}_{session}_{recording_number}_{class_code}
        # Example: f01_1_001_2 where 2=crying, 1=fussing, 0=neutral
        try:
            parts = filename.replace('.wav', '').replace('.mp3', '').split('_')
            if len(parts) >= 4:
                participant_id = parts[0].replace('f', '') if parts[0].startswith('f') else parts[0]
                session = parts[1] if parts[1].isdigit() else None
                recording_number = parts[2] if parts[2].isdigit() else None
                class_code = parts[3] if parts[3].isdigit() else None
                
                # Map class codes to labels
                class_mapping = {'0': 'neutral', '1': 'fussing', '2': 'crying'}
                class_from_code = class_mapping.get(class_code, class_code)
                
                return {
                    'participant_id': participant_id,
                    'session_number': int(session) if session else None,
                    'recording_number': int(recording_number) if recording_number else None,
                    'class_code': class_code,
                    'class_from_filename': class_from_code
                }
        except Exception as e:
            print(f"Error parsing CRIED filename {filename}: {e}")
        
        return {}
    
    def extract_baby_chilanto_metadata(self, filename):
        """Extract metadata from Baby Chilanto filename format"""
        # Formats vary: simple numbers (63, 11), descriptive (25b_nino_sordo, 52a)
        try:
            base_name = filename.replace('.wav', '').replace('.mp3', '')
            
            # Extract numeric ID if present
            numeric_id = None
            if base_name.isdigit():
                numeric_id = int(base_name)
            else:
                # Extract leading numbers
                import re
                match = re.match(r'(\d+)', base_name)
                if match:
                    numeric_id = int(match.group(1))
            
            # Check for special indicators
            is_child = 'nino' in base_name.lower() or 'nina' in base_name.lower()
            has_condition = any(word in base_name.lower() for word in ['sordo', 'deaf', 'pain', 'normal'])
            
            # Extract suffix (a, b, etc.)
            suffix = None
            import re
            suffix_match = re.search(r'(\d+)([a-z]+)', base_name)
            if suffix_match:
                suffix = suffix_match.group(2)
            
            return {
                'sample_id': numeric_id,
                'filename_base': base_name,
                'suffix': suffix,
                'is_child_indicated': is_child,
                'has_condition_indicator': has_condition,
                'original_filename': filename
            }
        except Exception as e:
            print(f"Error parsing Baby Chilanto filename {filename}: {e}")
        
        return {}
    
    def process_file(self, input_path, output_path, dataset_name, class_name, file_id):
        """Process a single audio file"""
        # Load and validate
        audio, original_sr, status = self.load_and_validate_audio(input_path)
        
        # Extract filename-based metadata
        filename = Path(input_path).name
        filename_metadata = {}
        
        if dataset_name == 'donateacry':
            filename_metadata = self.extract_donateacry_metadata(filename)
        elif dataset_name == 'cried':
            filename_metadata = self.extract_cried_metadata(filename)
        elif dataset_name == 'baby_chilanto':
            filename_metadata = self.extract_baby_chilanto_metadata(filename)
        
        if audio is None:
            return {
                'file_id': file_id,
                'dataset': dataset_name,
                'class': class_name,
                'original_filename': filename,
                'original_path': str(input_path),
                'status': status,
                'processed': False,
                **filename_metadata
            }
        
        # Calculate original metrics
        original_duration = len(audio) / original_sr
        quality_metrics = self.calculate_audio_quality_metrics(audio, original_sr)
        
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio, original_sr)
        processed_duration = len(processed_audio) / self.target_sr
        
        # Save preprocessed audio
        sf.write(output_path, processed_audio, self.target_sr)
        
        # Create comprehensive metadata entry
        metadata_entry = {
            'file_id': file_id,
            'dataset': dataset_name,
            'class': class_name,
            'original_filename': filename,
            'original_path': str(input_path),
            'processed_path': str(output_path),
            'original_sample_rate': original_sr,
            'target_sample_rate': self.target_sr,
            'original_duration': round(original_duration, 3),
            'processed_duration': round(processed_duration, 3),
            'duration_category': self.categorize_duration(original_duration),
            'status': status,
            'processed': True,
            **quality_metrics,
            **filename_metadata
        }
        
        return metadata_entry
    
    def process_dataset(self, dataset_path, dataset_name, class_structure):
        """
        Process entire dataset
        
        Args:
            dataset_path: Path to dataset root
            dataset_name: Name of dataset (donateacry, cried, baby_chilanto)
            class_structure: Dict mapping class names to their file paths or
                           directory structure info
        """
        print(f"\nProcessing {dataset_name} dataset...")
        
        processed_count = 0
        failed_count = 0
        
        for class_name, file_info in class_structure.items():
            print(f"  Processing class: {class_name}")
            
            # Handle different dataset structures
            if isinstance(file_info, list):
                # Direct file list
                file_paths = file_info
            else:
                # Directory structure - scan for audio files
                class_dir = Path(dataset_path) / class_name
                if class_dir.exists():
                    file_paths = list(class_dir.glob("*.wav")) + \
                                list(class_dir.glob("*.mp3")) + \
                                list(class_dir.glob("*.m4a"))
                else:
                    print(f"    Warning: Class directory not found: {class_dir}")
                    continue
            
            # Process each file in the class
            for i, file_path in enumerate(tqdm(file_paths, desc=f"    {class_name}")):
                file_id = f"{dataset_name}_{class_name}_{i:04d}"
                output_filename = f"{file_id}.wav"
                output_path = self.output_dir / dataset_name / output_filename
                
                metadata_entry = self.process_file(
                    file_path, output_path, dataset_name, class_name, file_id
                )
                
                self.metadata.append(metadata_entry)
                
                if metadata_entry['processed']:
                    processed_count += 1
                else:
                    failed_count += 1
        
        print(f"  {dataset_name}: {processed_count} files processed, {failed_count} failed")
        return processed_count, failed_count
    
    def save_metadata(self):
        """Save metadata to files"""
        # Save as JSON
        metadata_json_path = self.output_dir / "preprocessing_metadata.json"
        with open(metadata_json_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save as CSV for easy analysis
        metadata_csv_path = self.output_dir / "preprocessing_metadata.csv"
        df = pd.DataFrame(self.metadata)
        df.to_csv(metadata_csv_path, index=False)
        
        print(f"\nMetadata saved to:")
        print(f"  JSON: {metadata_json_path}")
        print(f"  CSV: {metadata_csv_path}")
        
        return df
    
    def generate_summary_report(self, metadata_df):
        """Generate preprocessing summary report"""
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY REPORT")
        print("="*50)
        
        # Overall statistics
        total_files = len(metadata_df)
        processed_files = len(metadata_df[metadata_df['processed'] == True])
        failed_files = total_files - processed_files
        
        print(f"\nOverall Statistics:")
        print(f"  Total files: {total_files}")
        print(f"  Successfully processed: {processed_files}")
        print(f"  Failed: {failed_files}")
        
        if processed_files > 0:
            processed_df = metadata_df[metadata_df['processed'] == True]
            
            # Dataset distribution
            print(f"\nDataset Distribution:")
            dataset_counts = processed_df['dataset'].value_counts()
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count} files")
            
            # Class distribution with additional metadata
            print(f"\nClass Distribution:")
            class_counts = processed_df.groupby(['dataset', 'class']).size()
            for (dataset, class_name), count in class_counts.items():
                print(f"  {dataset}/{class_name}: {count} files")
            
            # Dataset-specific metadata summaries
            if 'donateacry' in processed_df['dataset'].values:
                donateacry_df = processed_df[processed_df['dataset'] == 'donateacry']
                print(f"\nDonateACry Specific Metadata:")
                if 'gender' in donateacry_df.columns:
                    gender_dist = donateacry_df['gender'].value_counts()
                    print(f"  Gender distribution: {dict(gender_dist)}")
                if 'age_months' in donateacry_df.columns:
                    age_stats = donateacry_df['age_months'].describe()
                    print(f"  Age range: {age_stats['min']:.0f}-{age_stats['max']:.0f} months (mean: {age_stats['mean']:.1f})")
            
            if 'cried' in processed_df['dataset'].values:
                cried_df = processed_df[processed_df['dataset'] == 'cried']
                print(f"\nCRIED Specific Metadata:")
                if 'participant_id' in cried_df.columns:
                    unique_participants = cried_df['participant_id'].nunique()
                    print(f"  Unique participants: {unique_participants}")
                if 'session_number' in cried_df.columns:
                    session_dist = cried_df['session_number'].value_counts()
                    print(f"  Session distribution: {dict(session_dist)}")
            
            if 'baby_chilanto' in processed_df['dataset'].values:
                chilanto_df = processed_df[processed_df['dataset'] == 'baby_chilanto']
                print(f"\nBaby Chilanto Specific Metadata:")
                if 'sample_id' in chilanto_df.columns:
                    id_range = chilanto_df['sample_id'].describe()
                    print(f"  Sample ID range: {id_range['min']:.0f}-{id_range['max']:.0f}")
                if 'is_child_indicated' in chilanto_df.columns:
                    child_indicated = chilanto_df['is_child_indicated'].sum()
                    print(f"  Files with child indicators: {child_indicated}")
            
            
            # Duration statistics
            print(f"\nDuration Statistics:")
            print(f"  Min duration: {processed_df['original_duration'].min():.2f}s")
            print(f"  Max duration: {processed_df['original_duration'].max():.2f}s")
            print(f"  Mean duration: {processed_df['original_duration'].mean():.2f}s")
            print(f"  Median duration: {processed_df['original_duration'].median():.2f}s")
            
            # Duration categories
            print(f"\nDuration Categories:")
            duration_counts = processed_df['duration_category'].value_counts()
            for category, count in duration_counts.items():
                print(f"  {category}: {count} files")
            
            # Quality metrics
            print(f"\nQuality Metrics (averages):")
            print(f"  SNR estimate: {processed_df['snr_estimate_db'].mean():.2f} dB")
            print(f"  Dynamic range: {processed_df['dynamic_range_db'].mean():.2f} dB")
            print(f"  Zero crossing rate: {processed_df['zero_crossing_rate'].mean():.4f}")


# Example usage function
def main():
    """
    Main preprocessing pipeline
    Modify the paths and class structures according to your dataset organization
    """
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(target_sr=16000, output_dir="D:\TarunBali\infant_cry_knowledge_graph\preprocessed_audio")
    
    # Define your dataset structures
    # You need to modify these paths according to your actual dataset organization
    
    # Example for DonateACry dataset
    donateacry_structure = {
        'belly_pain': None,  # Add list of file paths or use directory scanning
        'burping': None,
        'discomfort': None,
        'hungry': None,
        'tired': None
    }
    
    # Example for CRIED dataset
    cried_structure = {
        'crying': None,
        'fussing': None,
        'neutral': None
    }
    
    # Example for Baby Chilanto dataset
    baby_chilanto_structure = {
        'asphyxia': None,
        'deaf': None,
        'hunger': None,
        'normal': None,
        'pain': None
    }
    
    # Process each dataset
    # Uncomment and modify paths as needed
    
    preprocessor.process_dataset("D:\TarunBali\infant_cry_knowledge_graph\donateacry", "donateacry", donateacry_structure)
    preprocessor.process_dataset("D:\TarunBali\infant_cry_knowledge_graph\cried", "cried", cried_structure)
    preprocessor.process_dataset(r"D:\TarunBali\infant_cry_knowledge_graph\baby_chilanto", "baby_chilanto", baby_chilanto_structure)
    
    # Save metadata and generate report
    metadata_df = preprocessor.save_metadata()
    preprocessor.generate_summary_report(metadata_df)
    
    print(f"\nPreprocessing complete! Check the '{preprocessor.output_dir}' directory for:")
    print("  - Preprocessed audio files (organized by dataset)")
    print("  - preprocessing_metadata.json (detailed metadata)")
    print("  - preprocessing_metadata.csv (for easy analysis)")


if __name__ == "__main__":
    main()