import os
import json
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class NSynthDataset(Dataset):
    def __init__(self, split='train', source_domains=None, target_domain=None,
                 max_samples_per_class=None, transform=None):
        """
        Args:
            split: 'train', 'valid', or 'test'
            source_domains: List of source domains to include (for training)
            target_domain: Target domain (for testing)
            max_samples_per_class: Maximum samples per class (for balancing)
            transform: Any transformations to apply to the MFCCs
        """
        self.split = split
        self.transform = transform
        self.base_path = f'/content/nsynth/nsynth-{split}'

        # Load metadata
        with open(f'{self.base_path}/examples.json', 'r') as f:
            self.metadata = json.load(f)

        # Convert to DataFrame
        self.df = pd.DataFrame.from_dict(self.metadata, orient='index')

        # Map instrument family to classes
        self.df['class'] = self.df['instrument_family_str']

        # Map source to domains
        self.df['domain'] = self.df['instrument_source_str']

        # Filter to target classes
        target_classes = ['bass', 'guitar', 'keyboard', 'mallet', 'vocal']
        self.df = self.df[self.df['class'].isin(target_classes)]

        # Filter by domains based on training or testing
        if source_domains and target_domain:
            if split == 'train':
                # For training, use only source domains
                self.df = self.df[self.df['domain'].isin(source_domains)]
            else:
                # For evaluation, use only target domain
                self.df = self.df[self.df['domain'] == target_domain]

        # Balance classes if needed
        if max_samples_per_class:
            balanced_data = []
            for class_name in target_classes:
                class_df = self.df[self.df['class'] == class_name]
                if len(class_df) > max_samples_per_class:
                    class_df = class_df.sample(max_samples_per_class, random_state=42)
                balanced_data.append(class_df)

            self.df = pd.concat(balanced_data)

        # Reset index
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={'index': 'note_id'})

        # Encode labels
        self.class_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()

        self.df['class_encoded'] = self.class_encoder.fit_transform(self.df['class'])
        self.df['domain_encoded'] = self.domain_encoder.fit_transform(self.df['domain'])

        self.num_classes = len(self.class_encoder.classes_)
        self.classes = self.class_encoder.classes_
        self.domains = self.domain_encoder.classes_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        note_id = self.df.loc[idx, 'note_id']
        audio_path = f'{self.base_path}/audio/{note_id}.wav'

        # Extract MFCC features
        mfcc = self.extract_mfcc(audio_path)

        # Get labels
        class_label = self.df.loc[idx, 'class_encoded']
        domain_label = self.df.loc[idx, 'domain_encoded']

        # Apply transforms if any
        if self.transform:
            mfcc = self.transform(mfcc)

        # Convert to tensor
        mfcc_tensor = torch.FloatTensor(mfcc)

        return mfcc_tensor, class_label, domain_label

    def extract_mfcc(self, audio_path, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCC features from audio file."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Transpose to have time as first dimension
        mfcc = mfcc.T

        # Make sure all MFCCs have the same length
        target_length = 130  # Similar to paper
        if mfcc.shape[0] < target_length:
            # Pad
            padding = np.zeros((target_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))
        elif mfcc.shape[0] > target_length:
            # Truncate
            mfcc = mfcc[:target_length, :]

        return mfcc