import time
import threading
import os
from flask import Flask, jsonify, render_template, request
import samna

# Import the extracted classes from utils
from utils import NeuralModel, EnergyMonitor, SensorMonitor

class NeuralMonitorApp:
    """
    Flask web application for neural network monitoring and comparison.
    
    Provides a web interface to:
    1. Load and run different neural models
    2. Compare SNN vs ANN performance
    3. Monitor energy consumption
    4. Stream simulated sensor data
    """
    def __init__(self):
        self.app = Flask(__name__)
        self.neural_model = NeuralModel()
        self.energy_monitor = EnergyMonitor()
        self.sensor_monitor = SensorMonitor()
        self.ann_result = None
        self.snn_result = None
        self._setup_routes()
        
    def _setup_routes(self) -> None:
        """Configure Flask URL routes and their corresponding handler methods"""
        self.app.route('/')(self.home)
        self.app.route('/energy', methods=['GET'])(self.get_energy_data)
        self.app.route('/sensor_data', methods=['GET'])(self.get_sensor_data)
        self.app.route('/load_model', methods=['POST'])(self.load_model)
        self.app.route('/start_inference', methods=['POST'])(self.start_inference)
        self.app.route('/start_baseline', methods=['POST'])(self.start_baseline)
        self.app.route('/get_baseline', methods=['GET'])(self.get_baseline)
        
    def home(self):
        """Render the main dashboard template"""
        return render_template('index.html')
    
    def get_energy_data(self):
        """API endpoint to fetch current energy readings"""
        return jsonify(self.energy_monitor.get_energy_data())
    
    def get_sensor_data(self):
        """API endpoint to fetch next sensor reading"""
        with self.app.app_context():
            return jsonify(self.sensor_monitor.get_sensor_data())
    
    def load_model(self):
        """
        API endpoint to load a neural model based on parameters.
        
        Expects a JSON payload with 'timesteps' and 'mode' parameters
        to determine which pre-trained model to load.
        
        Returns:
            JSON response with load status and model information
        """
        data = request.get_json()
        timesteps = data.get('timesteps')
        mode = data.get('mode')
        
        model_filename = (f'models/SynNet_kaggle_{timesteps}timesteps_20epochs.json' 
                         if mode == 'train' else 
                         f'models/converted_rp_model_20epochs_{timesteps}timesteps_kaggle.json')
        
        try:
            self.neural_model.create_model(mode=mode, model_path=model_filename)
            return jsonify({
                'status': 'success',
                'model': model_filename,
                'mode': mode,
                'timesteps': timesteps
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    def start_inference(self):
        """
        API endpoint to run inference on both ANN and SNN models.
        
        Runs inference in parallel threads to avoid blocking and compares
        performance metrics between the two neural architectures.
        
        Returns:
            JSON with inference results, including execution times and predictions
        """
        data = request.get_json()
        try:
            if self.neural_model.modSamna is None:
                return jsonify({
                    'status': 'error',
                    'message': 'No model loaded. Please load a model first.'
                }), 400
            
            # Create threads for parallel execution to avoid blocking
            ann_thread = threading.Thread(target=self._run_ann_inference)
            snn_thread = threading.Thread(target=self._run_snn_inference)
            
            # Start both threads
            ann_thread.start()
            snn_thread.start()
            
            # Wait for both threads to complete
            ann_thread.join()
            snn_thread.join()
            
            # Return combined results to frontend
            return jsonify({
                'status': 'success',
                'model': f"{data.get('mode')} model with {data.get('timesteps')} timesteps",
                'ann_output': str(self.ann_result['prediction']),
                'ann_time': self.ann_result['execution_time'],
                'snn_output': str(self.snn_result['prediction']),
                'snn_time': self.snn_result['execution_time']
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    def _run_ann_inference(self):
        """
        Run ANN inference in a separate thread and store results.
        
        Handles exceptions to ensure thread doesn't crash the application.
        """
        try:
            self.ann_result = self.neural_model.test_ann_performance()
        except Exception as e:
            self.ann_result = {
                'prediction': f"Error: {str(e)}",
                'execution_time': 0
            }

    def _run_snn_inference(self):
        """
        Run SNN inference in a separate thread and store results.
        
        Handles exceptions to ensure thread doesn't crash the application.
        """
        try:
            self.snn_result = self.neural_model.test_performance()
        except Exception as e:
            self.snn_result = {
                'prediction': f"Error: {str(e)}",
                'execution_time': 0
            }
        
    
    def start_baseline(self):
        """API endpoint to trigger baseline energy measurement"""
        try:
            result = self.energy_monitor.measure_baseline()
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    def get_baseline(self):
        """API endpoint to retrieve stored baseline energy value"""
        result = self.energy_monitor.get_baseline()
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)
    
    def run(self, host: str = '0.0.0.0', port: int = 5004) -> None:
        """
        Initialize and start the web application with monitoring threads.
        
        Cleans up any existing energy data file, starts monitoring threads
        for energy and sensor data, then launches the Flask application.
        
        Args:
            host: IP address to bind the server to
            port: Port number to listen on
        """
        # Remove any existing energy data from previous runs
        if os.path.exists('energy_data.csv'):
            os.remove('energy_data.csv')
            
        # Start monitoring threads
        energy_thread = threading.Thread(
             target=self.energy_monitor.start_monitoring,
             args=(self._get_xylo_device(),)
        )

        energy_thread.daemon = True
        
        sensor_thread = threading.Thread(
            target=self._run_sensor_monitoring
        )
        sensor_thread.daemon = True
        
        # Start threads and application
        energy_thread.start()
        time.sleep(2)  # Allow energy monitoring to initialize
        sensor_thread.start()
        
        self.app.run(host=host, port=port)
    
    def _get_xylo_device(self):
        """
        Find and connect to an available Xylo IMU device.
        
        Returns:
            Reference to the first available Xylo IMU device
        """
        device_list = samna.device.get_all_devices()
        imu_hdk_list = [
            samna.device.open_device(d)
            for d in device_list
            if d.device_type_name == "XyloImuTestBoard"
        ]
        return imu_hdk_list[0]
    
    def _run_sensor_monitoring(self):
        """Background thread that continuously simulates sensor data readings"""
        while True:
            self.get_sensor_data()

if __name__ == '__main__':
    app = NeuralMonitorApp()
    app.run()
