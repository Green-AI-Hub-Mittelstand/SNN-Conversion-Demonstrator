import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.networks import SynNet
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
from rockpool.devices.xylo import find_xylo_hdks
from rockpool.devices.xylo.syns63300 import (
    config_from_specification, 
    mapper
)
from rockpool.transform import quantize_methods as q
from utils.create_dataset import GearboxDataset
from torch.utils.data import DataLoader
from torchvision import transforms

class NeuralModel:
    """
    Manages neural network models for both training and inference.
    Handles both standard ANNs and spiking neural networks (SNNs),
    with hardware deployment to Xylo neuromorphic chips.
    """
    def __init__(self, dt: float = 0.001):
        # Default timestep of 1ms for temporal dynamics in spiking networks
        self.dt = dt
        self.model = None
        self.modSamna = None
        self.xylo_device = None

    def create_model(self, mode: str = "convert", model_path: Optional[str] = None, time_window: int = 50) -> Tuple[Any, Any]:
        """
        Creates or loads a neural network model based on specified parameters.
        
        Args:
            mode: "train" for creating a training model, "convert" for deploying to hardware
            model_path: Path to pre-trained model file (optional)
            time_window: Number of timesteps for temporal processing
            
        Returns:
            Tuple containing the Samna model and Xylo module
        """
        print("Loading model...")
        
        if mode == "train":
            # For training: Create a SynNet with specific architecture for gearbox fault detection
            self.model = SynNet(
                n_classes=2,
                n_channels=16,
                size_hidden_layers=[30],
                time_constants_per_layer=[5],
                tau_syn_base=0.002,
                tau_mem=Constant(0.002),
                neuron_model=LIFTorch,
                dt=self.dt,
            )
        else:
            # For conversion/inference: Create a Sequential model with LIF neurons for hardware deployment
            self.model = Sequential(
                LinearTorch((16, 30)),
                LIFTorch((30, 30)),
                LinearTorch((30, 16)),
                LIFTorch((16, 16)),
                LinearTorch((16, 2)),
                LIFTorch((2, 2))
            )

        # Load pre-trained weights from file
        if model_path:
            self.model.load(model_path)
        else:
            # Default model path when none specified
            self.model.load(f'models/converted_rp_model_20epochs_{time_window}timesteps_kaggle.json')

        # Map the model to Xylo hardware specification and quantize values
        spec = mapper(self.model.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')
        spec.update(q.channel_quantize(**spec))

        # Connect to physical Xylo device - fail if not found
        xylo_hdk_nodes, modules, versions = find_xylo_hdks()
        if not xylo_hdk_nodes or versions[0] != 'syns63300':
            raise RuntimeError('This application requires a connected Xylo IMU HDK.')
        
        db = xylo_hdk_nodes[0]
        x = modules[0]

        # Generate hardware configuration and verify it's valid
        config, is_valid, msg = config_from_specification(**spec)
        if not is_valid:
            raise RuntimeError(f"Invalid configuration: {msg}")

        # Create Samna interface to Xylo device with power monitoring at 20Hz
        self.modSamna = x.XyloSamna(db, config, dt=self.dt, power_frequency=20.)
        return self.modSamna, x

    def test_performance(self) -> dict:
        """
        Test SNN performance on hardware and measure execution time.
        
        Runs inference on 10 random samples from the test dataset and
        measures execution time to demonstrate SNN efficiency.
        
        Returns:
            Dictionary containing prediction results and execution time
        """
        path = 'archive'
        batch_size = 1
        
        print("loading test data")
        transform = transforms.Compose([])
        test_spike_data = GearboxDataset(
            path=path, 
            mode='test', 
            model_name="SNN", 
            transform=transform, 
            is_spiking=True, 
            time_window=50, 
            data_version='kaggle'
        )
        spike_test_loader = DataLoader(test_spike_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        print("starting test loop")
        # Start timing
        start_time = time.time()
        
        # Select 10 unique random indices for fair comparison
        unique_indices = np.random.choice(len(spike_test_loader.dataset), 10, replace=False)
        
        # Create a list for predictions
        predictions = []
        
        # Run inference on 10 unique samples
        for random_index in unique_indices:
            # Get the random sample
            random_inp, random_tgt = spike_test_loader.dataset[random_index]
            
            # Process the input - scale to integer values for hardware implementation
            random_inp = (np.array(random_inp.squeeze().tolist()) * 100).astype(int)
            
            # Run inference
            out, _, record_dict = self.modSamna(random_inp)
            
            # Store prediction
            predictions.append(out[-1])
        
        # Use the last prediction for return value
        out = predictions[-1]
        with torch.no_grad():
            pred = out[-1]
        
        # End timing
        execution_time = time.time() - start_time
        print(f"SNN inference time: {execution_time:.4f} seconds")
                
        print("finished test loop")
        return {
            'prediction': pred,
            'execution_time': execution_time
        }

    def test_ann_performance(self) -> dict:
        """
        Test performance of a standard ANN model (non-spiking) and measure execution time.
        
        Used for comparison with SNN performance to demonstrate efficiency differences
        between traditional ANNs and SNNs.
        
        Returns:
            Dictionary containing prediction output and execution time
        """
        path = 'archive'
        batch_size = 1
        
        print("loading ANN test data")
        transform = transforms.Compose([])
        test_data = GearboxDataset(
            path=path, 
            mode='test', 
            model_name="ANN", 
            transform=transform, 
            data_version='kaggle'
        )
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        print("starting ANN test loop")
        device = torch.device('cpu')

        # Create equivalent standard ANN architecture for comparison
        model = nn.Sequential(
            nn.Linear(16, 30, bias=True),
            nn.ReLU(),
            nn.Linear(30, 16, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2, bias=True),
            nn.ReLU()
        )  

        model.load_state_dict(torch.load('models/ann_test_kaggle.pth'))
        model.eval()

        
        # Start timing
        start_time = time.time()
        
        # Test on same number of samples as SNN for fair comparison
        unique_indices = np.random.choice(len(test_loader.dataset), 10, replace=False)

        # Create a list for predictions
        predictions = []

        # Run inference on 10 unique samples
        for random_index in unique_indices:
            # Get the random sample
            random_inp, random_tgt = test_loader.dataset[random_index]
            
            # Add batch dimension and move to device
            random_inp = random_inp.unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(random_inp)
                pred = output.argmax(dim=1)
                predictions.append(pred.item())

                    
        # End timing
        execution_time = time.time() - start_time
        print(f"ANN inference time: {execution_time:.4f} seconds")
                    
        return {
            'prediction': predictions[-1] if predictions else None,
            'execution_time': execution_time
        }
