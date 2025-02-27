from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tqdm
from scipy import stats


def preprocess_gearbox(path):
    """
    Preprocess raw gearbox sensor data for fault detection.
    
    This function:
    1. Reads multiple CSV files containing sensor readings
    2. Extracts information about gearbox state (healthy/faulty) from filenames
    3. Calculates statistical features from sensor readings
    4. Normalizes data to improve training performance
    
    Args:
        path: Directory containing raw gearbox sensor data files
        
    Returns:
        Tuple of (preprocessed_dataframe, labels_array)
    """
    dfs = []
    # Recursively walk through all files in the specified directory
    for dirname, _, filenames in tqdm.tqdm(os.walk(path)):
        for filename in tqdm.tqdm(filenames, leave=False):
            # Filter files: consider only those starting with 'h' (healthy) or 'b' (broken tooth)
            if filename[0] != "h" and filename[0] != "b":
                continue
                
            # Extract gearbox state from filename prefix
            state = filename[0]
            
            # Extract RPM (rotations per minute) from filename suffix
            rpm = int(filename.split('.')[0][5:])
            
            # Read sensor data from CSV file
            df = pd.read_csv(os.path.join(dirname, filename))
            
            # Add metadata columns
            df['state'] = state  # 'h' for healthy, 'b' for broken tooth
            df['load'] = rpm     # Motor speed in RPM
            
            # Collect all dataframes for later concatenation
            dfs.append(df)

    # Combine all individual dataframes into one large dataset
    df = pd.concat(dfs).reset_index().rename(columns={'index':'sample_index'})

    # Transform data to long format for easier grouping and feature extraction
    # Each row will represent one sensor reading with metadata
    sensor_readings = df.melt(
        id_vars=['sample_index','state', 'load'],
        value_vars=['a1','a2','a3','a4'],  # The four accelerometer channels
        var_name='sensor',
        value_name='reading'
    )

    # Commented code for saving raw sensor readings (useful for visualization)
    """sensor_readings_raw = sensor_readings.pivot(
        index=['sample_index', 'state', 'load'],  # Keep these columns as index
        columns='sensor',  # This will create new columns for each sensor
        values='reading'   # The values to fill in the new columns
    ).reset_index()  # Reset index to turn the multi-index DataFrame into a regular DataFrame

    # Optionally, rename the columns if needed
    sensor_readings_raw.columns.name = None  

    sensor_readings_raw.to_csv("dataset_raw.csv", index=False)"""

    # Initialize containers for extracted features
    data = {'a1':[], 'a2': [], 'a3':[]}
    labels = []
    
    # Group by state, load, and sensor type to process each series independently
    for (state,load,sensor),g in sensor_readings.groupby(['state','load','sensor']):
        # Skip a4 sensor as it's not needed for this analysis
        if sensor =='a4':
            continue
            
        # Get all readings for current sensor
        vals = g.reading.values
        
        # Split into fixed-length windows of 300 samples each
        # This creates segments for feature extraction
        splits = np.split(vals, range(300,vals.shape[0],300))
        
        for s in splits[:-1]:  # Skip the last window (might be incomplete)
            # For a1 sensor, include load (RPM) as an additional feature
            if sensor == 'a1':
                data[sensor].append([
                    float(load),             # Current RPM
                    np.mean(s),              # Mean amplitude
                    np.std(s),               # Standard deviation (signal energy)
                    stats.kurtosis(s),       # Kurtosis (peakedness)
                    stats.skew(s),           # Skewness (asymmetry)
                    stats.moment(s),         # First moment
                ])
            else:
                # For other sensors, only extract statistical features
                data[sensor].append([
                    np.mean(s),              # Mean amplitude
                    np.std(s),               # Standard deviation (signal energy)
                    stats.kurtosis(s),       # Kurtosis (peakedness)
                    stats.skew(s),           # Skewness (asymmetry)
                    stats.moment(s),         # First moment
                ])
                
            # Only create labels once per window (using a1 sensor as reference)
            if sensor == 'a1':
                labels.append(int(state=='b'))  # 1 for broken, 0 for healthy

    # Convert feature lists to dataframe and flatten the data structure
    df_data = pd.DataFrame(data)
    df_data = df_data.apply(lambda x: ','.join(x.astype(str)), axis=1)
    
    
    df_data = df_data.str.split(',', expand=True)
    df_data = df_data.apply(lambda x: x.str.strip('[]'))
    df_data = df_data.astype(float)
    data = df_data.values

    # Convert labels to numpy array for consistent processing
    labels = np.array(labels)
    
    # Normalize data using z-score standardization to improve neural network training
    # This ensures all features have similar ranges and zero mean
    means = data.mean(axis=0)
    stds = data.std(axis=0) +1e-6  # Add small epsilon to avoid division by zero
    data = (data - means) / stds

    # Create final dataframe with features and labels
    data = pd.DataFrame(data)
    data['label'] = labels

    # Save processed dataset to CSV for future use
    data.to_csv('dataset_kaggle.csv', index=False)

    return data, labels 


class GearboxDataset(Dataset):
    """
    PyTorch Dataset for gearbox fault detection data.
    
    Handles loading, preprocessing, and transforming gearbox sensor data
    for both standard neural networks (ANNs) and spiking neural networks (SNNs).
    
    Supports different data splitting modes (train/val/test) and
    transforms data into appropriate formats for different model architectures.
    """
    def __init__(self, model_name, path="", transform=None, mode='train', is_spiking=False, time_window=100, data=None, data_version=None):
        """
        Initialize the dataset with appropriate configuration.
        
        Args:
            model_name: Name of the model architecture being used
            path: Directory containing raw data files
            transform: PyTorch transforms to apply to data samples
            mode: 'train', 'val', 'test', 'all', or 'live' for different dataset splits
            is_spiking: True for spiking neural networks, False for standard ANNs
            time_window: Number of timesteps for temporal data (used for SNNs)
            data: Preloaded data (optional, otherwise loaded from path)
            data_version: Dataset version identifier ('kaggle' uses pre-processed data)
        """
        super().__init__()
        self.path = path
        if data_version == 'kaggle':
            # Load pre-processed data from CSV instead of raw files
            self.my_data = pd.read_csv('dataset_kaggle.csv')
        else:
            # Process raw data files on-the-fly
            self.my_data, self.targets = preprocess_gearbox(self.path)

        # Extract features and labels from dataframe
        self.data_raw = self.my_data.drop(columns=["label"]).to_numpy()
        self.targets = self.my_data["label"].to_numpy()

        # Shuffle data while preserving feature-label correspondence
        # This ensures random sampling but consistent feature-label pairs
        indices = np.arange(len(self.data_raw))
        np.random.seed(42)  # Fixed seed for reproducible experiments
        np.random.shuffle(indices)
        self.data_raw = self.data_raw[indices]
        self.targets = self.targets[indices]

        # Store configuration parameters
        self.transform = transform
        self.mode = mode
        self.is_spiking = is_spiking   
        self.time_window = time_window
        self.model_name = model_name
            
        # Convert numpy arrays to PyTorch tensors with appropriate types
        self.data = torch.tensor(self.data_raw).float()
        self.targets = torch.tensor(self.targets).float()

        # Convert targets to one-hot encoded vectors for classification
        self.targets = self.targets.to(torch.int64)
        self.targets = torch.nn.functional.one_hot(self.targets, 2)  # Binary classification (healthy/faulty)
        self.targets = self.targets.squeeze()
        
        # Calculate dataset split sizes (70% train, 10% validation, 20% test)
        self.train_split = int(len(self.data) * 0.7)
        self.val_split = int(len(self.data) * 0.1)
        self.test_split = int(len(self.data) * 0.2)
       
        # Select the appropriate subset based on mode
        if self.mode == 'train':
            self.data = self.data[:self.train_split]
            self.targets = self.targets[:self.train_split]
        elif self.mode == 'val':
            self.data = self.data[self.train_split:self.train_split+self.val_split]
            self.targets = self.targets[self.train_split:self.train_split+self.val_split]
        elif self.mode == 'test':
            self.data = self.data[self.train_split+self.val_split:]
            self.targets = self.targets[self.train_split+self.val_split:]
        elif self.mode == 'all' or self.mode == 'live':
            # Use all data (no subsetting)
            pass

        # For spiking networks, store the time window parameter
        if is_spiking:
            self.time_window = time_window

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data sample and its label.
        
        For spiking neural networks, transforms the data into a
        temporal sequence by repeating it across the time dimension.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (sample_tensor, label_tensor)
        """
        sample, label = self.data[idx], self.targets[idx]
        
        # Apply any transformations if specified
        if self.transform:
            sample = self.transform(sample)

        # Prepare data format for spiking neural networks
        if self.is_spiking:
            if self.model_name == 'conv1d':
                # Add channel dimension for 1D convolutions
                sample = sample.unsqueeze(0)

            # Repeat the sample across the time dimension to create temporal structure
            # This transforms static features into a time series for SNN processing
            sample = sample.repeat([self.time_window,1,1])

            # Alternative approach: Convert to binary spikes (commented out)
            #sample = (torch.rand(self.time_window, *sample.shape) < sample).float()

        else:
            # For standard ANNs with 1D convolutions, add channel dimension
            if self.model_name == 'conv1d':
                sample = sample.unsqueeze(0)
                
        return sample, label

