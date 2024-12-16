"""
Funzioni per la visualizzazione dei risultati del training.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_training_results(rewards, losses, window_size=100):
    """
    Visualizza i risultati del training.
    
    Args:
        rewards: Lista delle ricompense per episodio
        losses: Lista delle losses per episodio
        window_size: Dimensione della finestra per la media mobile
    """
    # Imposta lo stile
    sns.set_style("whitegrid")
    
    # Crea la figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot delle ricompense
    _plot_metric(ax1, rewards, "Ricompense", window_size, color='blue')
    
    # Plot delle losses
    _plot_metric(ax2, losses, "Loss", window_size, color='red')
    
    plt.tight_layout()
    plt.show()

def _plot_metric(ax, data, title, window_size, color):
    """Helper function per plottare una metrica con media mobile."""
    # Calcola la media mobile
    rolling_mean = pd.Series(data).rolling(window=window_size).mean()
    
    # Plot dei dati grezzi (più trasparente)
    ax.plot(data, alpha=0.3, color=color, label='Raw')
    
    # Plot della media mobile
    ax.plot(rolling_mean, color=color, label=f'Media Mobile ({window_size} ep.)')
    
    ax.set_title(title)
    ax.set_xlabel('Episodio')
    ax.set_ylabel(title)
    ax.legend()

def plot_q_values_heatmap(agent, title="Q-Values Distribution"):
    """
    Visualizza una heatmap dei valori Q dell'agente.
    
    Args:
        agent: L'agente con una Q-table
        title: Titolo del plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(agent.q_table, annot=True, cmap='viridis', fmt='.2f')
    plt.title(title)
    plt.xlabel('Azioni')
    plt.ylabel('Stati')
    plt.show()

def plot_episode_breakdown(rewards, actions, title="Analisi Episodio"):
    """
    Visualizza un breakdown dettagliato di un singolo episodio.
    
    Args:
        rewards: Lista delle ricompense dello step
        actions: Lista delle azioni intraprese
        title: Titolo del plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot delle ricompense
    ax1.plot(rewards, marker='o')
    ax1.set_title('Ricompense per Step')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Ricompensa')
    
    # Plot delle azioni
    ax2.plot(actions, marker='o', color='green')
    ax2.set_title('Azioni per Step')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Azione')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()