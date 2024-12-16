"""
Sistema di logging per tracciare metriche durante il training.
"""

import logging
import json
from datetime import datetime
import os

class Logger:
    def __init__(self, log_dir="logs"):
        """
        Inizializza il logger.
        
        Args:
            log_dir: Directory dove salvare i log
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup del logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Log su file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{timestamp}.log")
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Log su console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # Storage per metriche
        self.metrics = []
        
    def log(self, metrics_dict):
        """
        Logga un dizionario di metriche.
        
        Args:
            metrics_dict: Dizionario con le metriche da loggare
        """
        # Aggiungi timestamp
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        # Salva metriche
        self.metrics.append(metrics_dict)
        
        # Log formattato
        message = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) 
                            else f"{k}: {v}" for k, v in metrics_dict.items()])
        self.logger.info(message)
        
    def save_metrics(self):
        """Salva tutte le metriche in un file JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"metrics_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        self.logger.info(f"Metriche salvate in {filename}")