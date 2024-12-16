"""
Definizioni delle reti neurali utilizzate per gli agenti di deep reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        """
        Rete neurale per Deep Q-Learning.
        
        Args:
            input_dim: Dimensione dello spazio degli stati
            output_dim: Dimensione dello spazio delle azioni
            hidden_dims: Lista delle dimensioni dei layer nascosti
        """
        super(DQNetwork, self).__init__()
        
        # Costruzione dei layer
        layers = []
        prev_dim = input_dim
        
        # Crea i layer nascosti
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)  # Aggiunge regolarizzazione
            ])
            prev_dim = hidden_dim
        
        # Layer di output
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass della rete."""
        return self.network(x)
    
class DuelingDQNetwork(nn.Module):
    """
    Implementazione di una Dueling DQN che separa il calcolo del
    value state e dei vantaggi delle azioni.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQNetwork, self).__init__()
        
        # Feature extractor comune
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Stream per il value state
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Stream per i vantaggi delle azioni
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combina value e advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values 