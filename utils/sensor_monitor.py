import pandas as pd
from typing import Dict, Any

class SensorMonitor:
    """
    Simulates sensor data input stream from a dataset file.
    Provides sequential readings as if they were coming from live sensors.
    """
    def __init__(self):
        self.current_index = 0
        self.readings_list = []
        
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Retrieve the next sensor reading from the dataset.
        
        Increments through dataset rows sequentially and resets to beginning
        when finished, simulating a continuous sensor data stream.
        
        Returns:
            Dictionary containing sensor values or error if end of data reached
        """
        df = pd.read_csv('dataset_raw.csv')
        if self.current_index < len(df):
            latest_row = df.iloc[self.current_index]
            sensor_data = {
                'a1': latest_row['a1'],
                'a2': latest_row['a2'],
                'a3': latest_row['a3']
            }
            self.current_index += 1
            return sensor_data
        else:
            # Reset index to create a continuous loop of sensor data
            self.current_index = 0
            return {'error': 'End of data reached, resetting index.'}
