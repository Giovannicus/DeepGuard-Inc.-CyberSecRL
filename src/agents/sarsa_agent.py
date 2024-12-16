"""
Implementazione dell'agente SARSA (State-Action-Reward-State-Action).
SARSA è un algoritmo di apprendimento per rinforzo on-policy che aggiorna i valori Q
basandosi sulle azioni effettivamente intraprese dall'agente.
"""

import numpy as np
from typing import Tuple, Dict

class SARSAAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float, 
                 gamma: float, epsilon: float, epsilon_min: float, epsilon_decay: float):
        """
        Inizializza l'agente SARSA.
        
        Args:
            state_size: Dimensione dello spazio degli stati
            action_size: Dimensione dello spazio delle azioni
            learning_rate (α): Tasso di apprendimento - quanto velocemente l'agente aggiorna i suoi valori
            gamma (γ): Fattore di sconto - importanza delle ricompense future vs immediate
            epsilon (ε): Probabilità di esplorare vs sfruttare
            epsilon_min: Valore minimo di epsilon
            epsilon_decay: Tasso di decadimento di epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inizializza la Q-table con valori casuali piccoli
        self.q_table = np.random.uniform(low=-0.1, high=0.1, 
                                       size=(state_size, action_size))
        
    def get_action(self, state: int) -> int:
        """
        Seleziona un'azione usando la policy ε-greedy.
        Con probabilità ε sceglie un'azione casuale (esplorazione),
        altrimenti sceglie l'azione con il valore Q più alto (sfruttamento).
        """
        if np.random.random() < self.epsilon:
            # Esplorazione: sceglie un'azione casuale
            return np.random.randint(self.action_size)
        
        # Sfruttamento: sceglie l'azione con il valore Q più alto
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int) -> float:
        """
        Aggiorna la Q-table usando l'equazione di SARSA:
        Q(s,a) = Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
        
        Returns:
            loss: La loss dell'aggiornamento (differenza al quadrato)
        """
        # Valore Q corrente
        current_q = self.q_table[state, action]
        
        # Valore Q del prossimo stato-azione
        next_q = self.q_table[next_state, next_action]
        
        # Calcola il target usando l'equazione di SARSA
        target = reward + self.gamma * next_q
        
        # Aggiorna il valore Q
        self.q_table[state, action] += self.learning_rate * (target - current_q)
        
        # Calcola la loss
        loss = (target - current_q) ** 2
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def save(self, filepath: str) -> None:
        """Salva la Q-table su file."""
        np.save(filepath, self.q_table)
    
    def load(self, filepath: str) -> None:
        """Carica la Q-table da file."""
        self.q_table = np.load(filepath)