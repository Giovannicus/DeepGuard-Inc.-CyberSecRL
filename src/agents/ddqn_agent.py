"""
Implementazione dell'agente Double Deep Q-Network (DDQN).
DDQN è una versione migliorata del DQN che utilizza due reti neurali
per ridurre il sovrastima dei valori Q.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Rete neurale per l'approssimazione della Q-function.
        Usa tre layer fully connected con attivazioni ReLU.
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass della rete."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Buffer per memorizzare e campionare le esperienze passate."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Aggiunge una transizione al buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Campiona un batch di transizioni."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Converti in tensori PyTorch
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float,
                 gamma: float, epsilon: float, epsilon_min: float, 
                 epsilon_decay: float, memory_size: int, batch_size: int):
        """
        Inizializza l'agente DDQN.
        
        Args:
            state_size: Dimensione dello spazio degli stati
            action_size: Dimensione dello spazio delle azioni
            learning_rate: Tasso di apprendimento per l'ottimizzatore
            gamma: Fattore di sconto per ricompense future
            epsilon: Probabilità iniziale di esplorazione
            epsilon_min: Valore minimo di epsilon
            epsilon_decay: Tasso di decadimento di epsilon
            memory_size: Dimensione del replay buffer
            batch_size: Dimensione del batch per l'addestramento
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reti Q principale e target
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Ottimizzatore
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
    def get_action(self, state: np.ndarray) -> int:
        """
        Seleziona un'azione usando la policy ε-greedy.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self) -> float:
        """
        Aggiorna le reti usando un batch di esperienze.
        Implementa l'algoritmo Double DQN.
        
        Returns:
            loss: La loss dell'aggiornamento
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Campiona un batch di transizioni
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Calcola i valori Q correnti
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calcola i valori Q target usando Double DQN
        with torch.no_grad():
            # Seleziona le azioni usando la rete principale
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # Valuta le azioni usando la rete target
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Calcola la loss e aggiorna la rete
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Aggiorna la rete target copiando i pesi dalla rete principale."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str) -> None:
        """Salva i pesi della rete."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Carica i pesi della rete."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])