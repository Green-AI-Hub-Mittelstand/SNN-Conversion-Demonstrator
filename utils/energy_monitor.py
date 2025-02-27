import csv
import threading
import time
import numpy as np
from typing import List, Dict, Any
from carbontracking import MacEmmissionTracker
import samna
from rockpool.devices.xylo.syns63300 import xylo_imu_devkit_utils as hdkutils

class EnergyMonitor:
    """
    Monitors and records energy consumption of both the host system (Mac)
    and the Xylo neuromorphic device.
    
    Provides methods to establish baseline power consumption and track
    real-time energy usage for comparative analysis.
    """
    def __init__(self):
        self.energy_data: List[float] = []
        self.mac_tracker = MacEmmissionTracker(last_duration=1, co2=False)
        self.is_tracking = False
        self.mac_tracking_thread = None
        
    def measure_baseline(self, num_readings: int = 5) -> Dict[str, Any]:
        """
        Measure the baseline energy consumption of the system.
        
        Takes multiple readings to establish a reliable baseline when the
        system is idle. This value can later be subtracted from active
        measurements to isolate the energy cost of neural computations.
        
        Args:
            num_readings: Number of baseline readings to average
            
        Returns:
            Dictionary with baseline energy measurements in milliwatts
        """
        print("STARTING BASELINE MEASUREMENT")
        baseline_readings = []
        
        for _ in range(num_readings):
            reading = self.mac_tracker.flush()
            energy = (reading['AppleSiliconChip (Apple M3 Pro > CPU)'] + 
                     reading['AppleSiliconChip (Apple M3 Pro > GPU)'])
            baseline_readings.append(energy)
            
        baseline_average = sum(baseline_readings) / len(baseline_readings)
        print(f"Baseline average: {baseline_average}")
        
        self._save_baseline(baseline_average)
        return {
            'status': 'success',
            'baseline': baseline_average * 1000,  # Convert to milliwatts for UI consistency
            'readings': [r * 1000 for r in baseline_readings]
        }
    
    def _save_baseline(self, baseline: float) -> None:
        """
        Save the baseline energy measurement to disk for future reference.
        
        Args:
            baseline: The measured baseline in watts
        """
        with open('baseline_energy.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['baseline_energy'])
            writer.writerow([baseline * 1000])  # Store in milliwatts
    
    def get_baseline(self) -> Dict[str, Any]:
        """
        Retrieve the stored baseline energy measurement.
        
        Returns:
            Dictionary with baseline value or error message if not found
        """
        try:
            with open('baseline_energy.csv', 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                baseline = float(next(reader)[0])
                return {'baseline': baseline}
        except FileNotFoundError:
            return {
                'status': 'error',
                'message': 'No baseline has been established'
            }

    def start_monitoring(self, device) -> None:
        """
        Start concurrent energy monitoring for both Xylo device and Mac CPU/GPU.
        
        Spawns separate threads for each monitoring task to enable real-time
        data collection without impacting performance measurements.
        
        Args:
            device: Reference to the connected Xylo device
        """
        print("Starting energy monitoring...")
        # Start Xylo monitoring
        power_frequency = 1  # 1Hz sampling rate
        power_buf, power = hdkutils.set_power_measure(device, power_frequency)
        power.start_auto_power_measurement(power_frequency)

        # Start Xylo monitoring thread
        xylo_thread = threading.Thread(
            target=self._monitor_xylo,
            args=(power_buf, power)
        )
        xylo_thread.daemon = True
        xylo_thread.start()

        # Start Mac monitoring thread
        self.is_tracking = True
        self.mac_tracking_thread = threading.Thread(target=self._monitor_mac)
        self.mac_tracking_thread.daemon = True
        self.mac_tracking_thread.start()
    
    def _monitor_xylo(self, power_buf, power) -> None:
        """
        Background thread for monitoring Xylo device power consumption.
        
        Continuously reads from power buffer and logs core and I/O power.
        
        Args:
            power_buf: Buffer storing power measurements
            power: Power measurement interface
        """
        while True:
            ps = power_buf.get_events()
            
            channels = samna.xyloImuBoards.MeasurementChannels
            io_power = np.array([e.value for e in ps if e.channel == int(channels.Io)])
            core_power = np.array([e.value for e in ps if e.channel == int(channels.Core)])
            
            # Calculate total power (core + I/O)
            energy_value = core_power.mean() + io_power.mean()
            self._update_energy_data(energy_value*100, 'xylo')  # Convert to milliwatts
            time.sleep(1)
    
    def _monitor_mac(self) -> None:
        """
        Background thread for monitoring Mac CPU/GPU power consumption.
        
        Polls the MacEmmissionTracker at regular intervals to record host
        system energy usage during model inference.
        """
        print("Starting Mac energy monitoring...")
        while self.is_tracking:
            try:
                reading = self.mac_tracker.flush()
                energy = (reading['AppleSiliconChip (Apple M3 Pro > CPU)'] + 
                         reading['AppleSiliconChip (Apple M3 Pro > GPU)'])
                self._update_energy_data(energy * 1000000, 'mac')  # Convert to milliwatts
                
                print(f"Mac energy: {energy} kWh")
            except Exception as e:
                print(f"Error in Mac energy monitoring: {e}")


    def _update_energy_data(self, energy_value: float, source: str) -> None:
        """
        Record energy measurement to CSV for later analysis.
        
        Args:
            energy_value: Power consumption in milliwatts
            source: 'xylo' or 'mac' to identify measurement source
        """
        with open('energy_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([energy_value, source])
    
    def get_energy_data(self) -> Dict[str, Any]:
        """
        Retrieve the most recent energy readings for both sources.
        
        Returns:
            Dictionary with latest Xylo and Mac energy measurements
        """
        try:
            with open('energy_data.csv', mode='r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                if not rows:
                    return {'xylo': 0, 'mac': 0}
                
                # Get latest readings for each source (scan backwards for efficiency)
                xylo_reading = next((float(row[0]) for row in reversed(rows) 
                                   if row[1] == 'xylo'), 0)
                mac_reading = next((float(row[0]) for row in reversed(rows) 
                                  if row[1] == 'mac'), 0)
                
                return {
                    'xylo': xylo_reading,
                    'mac': mac_reading
                }
        except FileNotFoundError:
            return {'xylo': 0, 'mac': 0}
