"""
Wrapper per l'ambiente gym-idsgame che semplifica l'interfaccia e aggiunge funzionalità utili.
"""

import gym
import numpy as np

class IDSGameWrapper:
    def __init__(self, attack_type="random"):
        """
        Inizializza l'ambiente gym-idsgame con configurazioni specifiche.
        
        Args:
            attack_type: Tipo di attacco ('random' o 'maximal')
        """
        # Scegli l'ambiente appropriato in base al tipo di attacco
        env_name = f"idsgame-{attack_type}-attack-v0"
        
        # Configurazione base dell'ambiente
        self.env_config = {
            "num_layers": 3,
            "num_servers_per_layer": 3,
            "random_seed": 42,
            "local_view_observations": False
        }
        
        self.env = gym.make(env_name, **self.env_config)
        
        # Cache delle dimensioni per facile accesso
        self.state_dim = self.env.observation_space.n
        self.action_dim = self.env.action_space.n
        
    def reset(self):
        """Resetta l'ambiente e ritorna lo stato iniziale."""
        return self.env.reset()
    
    def step(self, action):
        """
        Esegue un'azione nell'ambiente.
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        next_state, reward, done, info = self.env.step(action)
        
        # Modifica la reward per incentivare comportamenti desiderati
        reward = self._modify_reward(reward, info)
        
        return next_state, reward, done, info
    
    def _modify_reward(self, reward, info):
        """
        Modifica la reward per migliorare l'apprendimento.
        Aggiunge bonus/penalità basati su eventi specifici.
        """
        modified_reward = reward
        
        # Bonus per prevenire attacchi riusciti
        if info.get('attack_prevented', False):
            modified_reward += 2.0
            
        # Penalità per falsi positivi
        if info.get('false_positive', False):
            modified_reward -= 1.0
            
        return modified_reward
    
    def close(self):
        """Chiude l'ambiente."""
        self.env.close()