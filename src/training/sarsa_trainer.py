"""
Classe per gestire l'addestramento dell'agente SARSA.
"""

import numpy as np
from tqdm import tqdm
from src.utils.logger import Logger
from src.utils.visualization import plot_training_results

class SARSATrainer:
    def __init__(self, agent, env, config, logger=None):
        """
        Inizializza il trainer.
        
        Args:
            agent: Istanza dell'agente SARSA
            env: Ambiente di training
            config: Configurazione del training
            logger: Logger per tracciare metriche
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger or Logger()
        
        # Storage per le metriche
        self.rewards_history = []
        self.losses_history = []
        
    def train(self):
        """Esegue il training completo."""
        print("Iniziando il training...")
        
        for episode in tqdm(range(self.config.num_episodes)):
            episode_reward, episode_loss = self._train_episode()
            
            # Logging
            self.rewards_history.append(episode_reward)
            self.losses_history.append(episode_loss)
            
            # Log periodico
            if (episode + 1) % self.config.log_interval == 0:
                self._log_progress(episode)
                
        # Plotta i risultati finali
        self._plot_results()
        
        return self.rewards_history, self.losses_history
    
    def _train_episode(self):
        """Esegue un singolo episodio di training."""
        state = self.env.reset()
        action = self.agent.get_action(state)
        total_reward = 0
        losses = []
        
        for step in range(self.config.max_steps):
            # Esegui azione
            next_state, reward, done, _ = self.env.step(action)
            next_action = self.agent.get_action(next_state)
            
            # Aggiorna l'agente
            loss = self.agent.update(state, action, reward, next_state, next_action)
            
            # Aggiorna statistiche
            total_reward += reward
            losses.append(loss)
            
            # Preparati per il prossimo step
            state = next_state
            action = next_action
            
            if done:
                break
                
        return total_reward, np.mean(losses)
    
    def _log_progress(self, episode):
        """Logga il progresso del training."""
        recent_rewards = self.rewards_history[-100:]
        self.logger.log({
            'episode': episode + 1,
            'avg_reward': np.mean(recent_rewards),
            'epsilon': self.agent.epsilon,
            'recent_loss': self.losses_history[-1]
        })
        
    def _plot_results(self):
        """Visualizza i risultati del training."""
        plot_training_results(
            rewards=self.rewards_history,
            losses=self.losses_history,
            window_size=100
        )