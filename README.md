<a name="readme-top"></a>

<br />
<div align="center">
  <h1 align="center">SNN Conversion Demonstrator</h1>
  
  <p align="center">
    <a href="https://github.com/Green-AI-Hub-Mittelstand/readme_template/issues">Report Bug</a>
    ·
    <a href="https://github.com/Green-AI-Hub-Mittelstand/readme_template/issues">Request Feature</a>
  </p>

  <br />

  <p align="center">
    <a href="https://www.green-ai-hub.de">
    <img src="images/green-ai-hub-keyvisual.svg" alt="Logo" width="80%">
  </a>
    <br />
    <h3 align="center"><strong>Green-AI Hub Mittelstand</strong></h3>
    <a href="https://www.green-ai-hub.de"><u>Homepage</u></a> 
    | 
    <a href="https://www.green-ai-hub.de/kontakt"><u>Contact</u></a>
  
   
  </p>
</div>

<br/>

## Overview

This project provides an interactive dashboard for loading neural models, running inference on both SNN and ANN architectures, monitoring energy consumption, and streaming simulated sensor data. It's designed to showcase the benefits and trade-offs between traditional neural networks and neuromorphic computing approaches.

## Features

- **Neural Model Comparison**: Run and compare inference between SNN and ANN models
- **Real-time Energy Monitoring**: Track energy consumption during model execution
- **Sensor Data Simulation**: Stream simulated sensor readings for testing
- **Interactive Web Dashboard**: Control experiments through an easy-to-use interface
- **Parallel Execution**: Run ANN and SNN inference simultaneously for direct comparison

## Installation

### Prerequisites

- Python 3.7+
- Flask
- Samna (neuromorphic computing library)
- Access to Xylo IMU hardware for energy monitoring

### Setup

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/SNN-Conversion-Demonstrator.git
   cd SNN-Conversion-Demonstrator
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary models in the `models/` directory.

## Usage

1. Start the application:

   ```
   python app.py
   ```

2. Access the web interface at `http://localhost:5004`

3. Using the interface:
   - Load a model with your chosen parameters
   - Run inference to compare SNN vs ANN performance
   - Monitor energy consumption during execution
   - View simulated sensor data

## Architecture

The application is structured around several key components:

- **NeuralMonitorApp**: Main Flask application that coordinates all components
- **NeuralModel**: Handles loading and running both SNN and ANN models
- **EnergyMonitor**: Tracks power consumption on neuromorphic hardware
- **SensorMonitor**: Provides simulated sensor data for testing

## API Endpoints

- `GET /`: Main dashboard interface
- `GET /energy`: Get current energy readings
- `GET /sensor_data`: Get next sensor reading
- `POST /load_model`: Load a neural model with specified parameters
- `POST /start_inference`: Run inference on both ANN and SNN models
- `POST /start_baseline`: Trigger baseline energy measurement
- `GET /get_baseline`: Retrieve stored baseline energy value

## Model Support

The application supports different neural network models:

- **Training Mode**: Uses `models/SynNet_kaggle_{timesteps}timesteps_20epochs.json`
- **Inference Mode**: Uses `models/converted_rp_model_20epochs_{timesteps}timesteps_kaggle.json`

## Hardware Requirements

For full functionality, you'll need access to:

- Xylo IMU TestBoard for accurate energy measurements
- Sufficient compute resources for neural network inference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
