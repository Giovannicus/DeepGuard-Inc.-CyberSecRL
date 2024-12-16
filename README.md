# Healthcare Cybersecurity with Reinforcement Learning

Un progetto di sicurezza informatica che utilizza algoritmi di Reinforcement Learning per proteggere le reti sanitarie attraverso simulazioni di attacco e difesa.

## 🎯 Obiettivi del Progetto

- Implementazione dell'algoritmo SARSA per scenari di "random attack"
- Implementazione dell'algoritmo DDQN per scenari di "random attack" e "maximal attack"
- Utilizzo dell'ambiente gym-idsgame per simulazioni di sicurezza
- Analisi e visualizzazione dei risultati

## 🛠️ Requisiti

- Python 3.8+
- PyTorch
- gym-idsgame
- Altri requisiti in `requirements.txt`

## 🚀 Installazione

```bash
# Clona il repository
git clone https://github.com/tuouser/healthcare-cybersec-rl.git
cd healthcare-cybersec-rl

# Crea un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
.\venv\Scripts\activate  # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

## 📝 Struttura del Progetto

- `notebooks/`: Jupyter notebooks con implementazioni SARSA e DDQN
- `src/`: Codice sorgente del progetto
- `tests/`: Test unitari
- `data/`: Directory per modelli salvati e risultati

## 🔬 Esperimenti

### SARSA Implementation
- Notebook: `notebooks/01_SARSA_RandomAttack.ipynb`
- Scenario: Random Attack
- Metriche di valutazione e risultati

### DDQN Implementation
- Notebook: `notebooks/02_DDQN_Attacks.ipynb`
- Scenari: Random Attack e Maximal Attack
- Confronto delle performance

## 📊 Risultati

[Inserire screenshots e grafici dei risultati principali]

## 🤝 Contributing

Le pull request sono benvenute. Per modifiche importanti, apri prima un issue per discutere cosa vorresti cambiare.

## 📄 License

[MIT License](LICENSE)