# 🚀 Quick Start Guide

Get up and running with DQN LunarLander in 5 minutes!

## ⚡ TL;DR

```bash
# Clone and setup
git clone https://github.com/SaintJeane/lunar-lander-dqn.git
cd lunar-lander-dqn
./setup.sh

# Train (2000 episodes, ~1-2 hours)
python train.py

# Evaluate
python evaluate.py --model models/best_model.pth --test --render
```

## 📋 Prerequisites

- **Python 3.8+** (3.10 recommended)
- **4GB RAM** minimum (8GB recommended)
- **CUDA GPU** (optional, but 10x faster training)
- **Linux/Mac/Windows** (all supported)

## 🎯 Installation Options

### Option 1: Local Installation (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/SaintJeane/lunar-lander-dqn.git
cd lunar-lander-dqn

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Option 2: Docker

```bash
# Build image
docker-compose build

# Train agent
docker-compose up dqn-training
```

### Option 3: Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch gymnasium[box2d] numpy matplotlib

# Create directories
mkdir -p models plots videos logs
```

## 🏃 Running the Project

### 1. Training

**Default training** (2000 episodes):
```bash
python train.py
```

**Custom training**:
```bash
# Quick test (100 episodes)
python -c "from train import train_dqn; train_dqn(num_episodes=100)"

# Extended training (5000 episodes)
python -c "from train import train_dqn; train_dqn(num_episodes=5000)"
```

**Monitor training**:
```bash
# Watch training in real-time
tail -f logs/training.log

# Check plots
ls plots/
```

### 2. Evaluation

**Standard evaluation** (100 episodes):
```bash
python evaluate.py --model models/best_model.pth --episodes 100
```

**Visual test** (single episode with rendering):
```bash
python evaluate.py --model models/best_model.pth --test --render
```

**Record videos**:
```bash
python evaluate.py --model models/best_model.pth --episodes 10 --record
```

### 3. Analysis

**Run Jupyter notebook**:
```bash
jupyter notebook analysis.ipynb
```

**Generate analysis plots**:
```bash
python -c "from analysis import *"
```

## 📊 Expected Results

### Training Progress

You should see output like:

```
Episode 100/2000 | Reward: -150.23 | Avg Reward: -180.45 | Epsilon: 0.606 | Loss: 0.0234
Episode 200/2000 | Reward: -85.67 | Avg Reward: -120.34 | Epsilon: 0.367 | Loss: 0.0189
Episode 500/2000 | Reward: 120.45 | Avg Reward: 95.23 | Epsilon: 0.081 | Loss: 0.0145
Episode 1000/2000 | Reward: 245.67 | Avg Reward: 220.12 | Epsilon: 0.010 | Loss: 0.0098
New best average reward: 220.12 - Model saved!
```

### Success Criteria

✅ **Solved**: Average reward ≥ 200 over 100 episodes  
✅ **Good**: Average reward ≥ 250  
✅ **Excellent**: Average reward ≥ 280

### Timeline

- **Episodes 0-500**: Learning basic control (negative rewards)
- **Episodes 500-1000**: Rapid improvement (reaching 100-200)
- **Episodes 1000-1500**: Mastering landings (200+)
- **Episodes 1500-2000**: Optimization (250+)

## 🐛 Troubleshooting

### Problem: Training is very slow

**Solution**:
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA version of PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "Box2D not found" error

**Solution**:
```bash
# Install Box2D dependencies
pip install gymnasium[box2d]

# Or manually
pip install box2d-py swig
```

### Problem: Out of memory error

**Solution**:
```python
# Reduce buffer size in dqn_agent.py
buffer_capacity = 50000  # Instead of 100000

# Or reduce batch size
batch_size = 32  # Instead of 64
```

### Problem: Training doesn't converge

**Possible causes and solutions**:

1. **Learning rate too high**:
   ```python
   learning_rate = 5e-4  # Reduce from 1e-3
   ```

2. **Epsilon decays too fast**:
   ```python
   epsilon_decay = 0.998  # Increase from 0.995
   ```

3. **Network too small**:
   ```python
   hidden_dim = 256  # Increase from 128
   ```

## 💡 Pro Tips

### Speed Up Training

1. **Use GPU**: 10x faster
   ```bash
   # Check GPU
   nvidia-smi
   ```

2. **Reduce logging**: Less frequent printing
   ```python
   log_interval = 50  # Instead of 10
   ```

3. **Pre-fill buffer**: Start with random experiences
   ```python
   # Add to train.py
   for _ in range(10000):
       state, _ = env.reset()
       action = env.action_space.sample()
       next_state, reward, done, _, _ = env.step(action)
       agent.replay_buffer.push(state, action, reward, next_state, done)
   ```

### Better Results

1. **Tune hyperparameters**: See ARCHITECTURE.md
2. **Longer training**: 3000-5000 episodes
3. **Multiple runs**: Take best of 3-5 runs

### Visualize Training

1. **TensorBoard** (requires installation):
   ```bash
   pip install tensorboard
   tensorboard --logdir=logs/
   ```

2. **Real-time plotting**:
   ```python
   # Add to train.py
   import matplotlib.pyplot as plt
   plt.ion()
   fig, ax = plt.subplots()
   ax.plot(episode_rewards)
   plt.pause(0.01)
   ```

## 📚 Next Steps

### Beginner

1. ✅ Run default training
2. ✅ Evaluate and visualize
3. 📖 Read README.md
4. 🔬 Experiment with hyperparameters

### Intermediate

1. 📊 Analyze results in Jupyter notebook
2. 🔧 Implement Double DQN
3. 📈 Compare with other algorithms
4. 🐛 Contribute bug fixes

### Advanced

1. 🌈 Implement Rainbow DQN
2. 🚀 Optimize for production
3. 📦 Deploy as API
4. 📝 Write blog post about your experience

## 🤔 FAQ

**Q: How long does training take?**  
A: 1-2 hours on GPU, 10-20 hours on CPU for 2000 episodes.

**Q: Can I pause and resume training?**  
A: Yes, load the checkpoint and continue:
```python
agent.load('models/checkpoint_ep1000.pth')
# Continue training from episode 1001
```

**Q: How much does the randomness affect results?**  
A: Significantly. Run 3-5 times and average results.

**Q: Can I use this on other environments?**  
A: Yes! Just change the environment name and adjust state/action dimensions.

**Q: What if I don't have a GPU?**  
A: It will work on CPU, just slower. Consider using Google Colab for free GPU.

## 🆘 Getting Help

- **Documentation**: Check README.md, ARCHITECTURE.md, CONTRIBUTING.md
- **Email**: Contact [maintainer](saintpetre99@gmail.com)

## 📝 Quick Reference

### Key Files

```
dqn_agent.py    - DQN implementation
train.py        - Training script
evaluate.py     - Evaluation script
config.py       - Configuration
test_dqn.py     - Unit tests
```

### Key Commands

```bash
make install    - Install dependencies
make train      - Train agent
make evaluate   - Evaluate agent
make test       - Run tests
make clean      - Remove generated files
```

### Key Hyperparameters

```python
learning_rate = 1e-3
gamma = 0.99
epsilon_decay = 0.995
batch_size = 64
buffer_capacity = 100000
```

---

🎉 **You're ready to go!** Start with `python train.py` and watch your agent learn to land!

For detailed documentation, see [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).