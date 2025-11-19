# Quantum Reinforcement Learning for Cryptocurrency Portfolio Management
## Hybrid PPO + QAC (Variational Quantum Circuit) Framework for Multi-Asset Crypto Allocation

---

## Overview

This project investigates whether Quantum Reinforcement Learning (QRL) can enhance cryptocurrency portfolio management relative to classical Proximal Policy Optimization (PPO).  
The system uses hourly Binance OHLCV data, computes technical indicators, simulates realistic market microstructure, and trains two agents:

- **Classical RL Agent:** PPO with Transformer Encoder  
- **Quantum RL Agent:** Quantum Actor–Critic (QAC) using Variational Quantum Circuits (VQC)

A Streamlit web application is provided for:

- Visualization of training progress  
- Evaluation of quantum circuit activations  
- Live allocation using real-time market data  
- On-the-fly fine-tuning  

This project demonstrates how hybrid quantum–classical models may improve exploration efficiency and risk-adjusted returns in volatile crypto markets.

---

## Features

### 1. Data and Feature Engineering
- Hourly OHLCV data from Binance  
- Technical indicators: RSI, MACD, Bollinger Bands, volatility, moving averages, log-returns, volume metrics  
- Sliding-window temporal representation  

### 2. Microstructure Environment
Includes realistic components:
- Slippage  
- Temporary and permanent market impact  
- Participation-rate constraints  
- Liquidity and execution noise  

Reward function:

\[
r = \text{Return} - \lambda \cdot \text{Volatility} - \eta \cdot \text{Turnover}
\]

### 3. PPO Agent (Classical)
- Transformer encoder  
- Actor–Critic architecture  
- PPO clipping for stable learning  

### 4. Quantum RL Agent (QAC)
- Classical feature encoding via angle embedding  
- Variational Quantum Circuit constructed using PennyLane  
- Quantum Actor–Critic for continuous portfolio weights  
- Enhanced exploration via quantum superposition and entanglement  

### 5. Streamlit Dashboard
- Training analytics  
- Cumulative returns and Sharpe ratios  
- Qubit activation and entanglement maps  
- Live allocation and fine-tuning  

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/sanjana-datascience-ai/Quantum-Reinforcement-Learning-for-Cryptocurrency-Portfolio-Management.git
cd Quantum-Reinforcement-Learning-for-Cryptocurrency-Portfolio-Management
```


### 2. Create a Python environment
```bash
conda create -n qrl python=3.10
conda activate qrl
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
---

## Running the System

### Load the data
```bash
python loader.py
```

### Computing Indicators
```bash
python features.py
```

### Train PPO Agent
```bash
python ppo_runner.py
```

### Train Quantum RL (QAC) Agent
```bash
python qrl_3assets_runner.py
```
### Comparison
```bash
python comparison.py
```

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```

### Results Summary
| Model     | Cumulative Return | Sharpe Ratio | Notes                                  |
| --------- | ----------------- | ------------ | -------------------------------------- |
| PPO       | ~8.54%            | 0.46         | Stable but slower adaptation           |
| QRL (QAC) | ~119.83%          | 1.21         | Strong improvement; better exploration |


### Key observations:
* Approximately 14× higher returns using QRL
* Nearly 3× higher Sharpe ratio
* More efficient exploration and quicker regime adaptation

### Limitations
* Quantum circuit simulation is computationally intensive
* Variational quantum models scale poorly in asset dimensionality
* Training more than thress assets becomes memory-heavy on classical hardware
* Off-policy algorithms like SAC/TD3 struggle with simplex-constrained outputs
* No real quantum hardware used; all results rely on classical simulation

### Future Work
* Deployment on real IBM/Qiskit QPUs
* Expanding to 8–10 asset portfolios with optimized circuits
* Incorporating order-book depth into the environment
* Exploring hybrid transformer–quantum architectures
* Integration with real-time automated trading systems

#Author
Sanjana R
B.Tech (Hons) — Data Science
Vidyashilp University, Bengaluru
