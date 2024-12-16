"""Configuration settings for the project."""

from dataclasses import dataclass
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SARSAConfig:
    """Configuration for SARSA agent."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    episodes: int = 1000
    max_steps: int = 500
    
@dataclass
class DDQNConfig:
    """Configuration for DDQN agent."""
    learning_rate: float = 0.0001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update: int = 10
    episodes: int = 1000
    max_steps: int = 500

class ProjectConfig:
    """Main configuration class for the project."""
    
    def __init__(self):
        self.sarsa = SARSAConfig()
        self.ddqn = DDQNConfig()
        
        # Paths
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODELS_DIR = os.path.join(self.DATA_DIR, "models")
        self.RESULTS_DIR = os.path.join(self.DATA_DIR, "results")
        
        # Environment settings
        self.env_config = {
            "num_layers": 3,
            "num_servers_per_layer": 3,
            "random_seed": 42,
            "local_view_observations": False,
        }
        
        # Create directories if they don't exist
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
    
    def get_env_config(self) -> Dict[str, Any]:
        """Returns the environment configuration."""
        return self.env_config.copy()

# Create global config instance
config = ProjectConfig()