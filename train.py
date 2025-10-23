# train.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os
import json
from datetime import datetime

def train_dqn(
    num_episodes=2000,
    max_steps=1000,
    save_dir="./models",
    plot_dir="./plots",
    log_interval=10
):
    """
    Train DQN agent on LunarLander-v2 environment.
    
    Args:
        num_episodes: Total number of training episodes
        max_steps: Maximum steps per episode
        save_dir: Directory to save model checkpoints
        plot_dir: Directory to save training plots
        log_interval: Episodes between progress logs
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    moving_avg_rewards = []
    best_avg_reward = -float('inf')
    
    print("=" * 60)
    print("Starting DQN Training on LunarLander-v2")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print(f"Initial Epsilon: {agent.epsilon}")
    print(f"Batch Size: {agent.batch_size}")
    print(f"Gamma: {agent.gamma}")
    print("=" * 60)
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # Calculate moving average (last 100 episodes)
        window_size = min(100, episode)
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)
        
        # Log progress
        if episode % log_interval == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (100ep): {moving_avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Save best model
        if moving_avg > best_avg_reward and episode > 100:
            best_avg_reward = moving_avg
            agent.save(os.path.join(save_dir, "best_model.pth"))
            print(f"New best average reward: {best_avg_reward:.2f} - Model saved!")
        
        # Save checkpoint every 500 episodes
        if episode % 500 == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pth"))
    
    # Save final model
    agent.save(os.path.join(save_dir, "final_model.pth"))
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'moving_avg_rewards': moving_avg_rewards,
        'best_avg_reward': best_avg_reward,
        'training_date': datetime.now().isoformat()
    }
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    plot_training_results(episode_rewards, moving_avg_rewards, episode_losses, plot_dir)
    
    env.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Models saved to: {save_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("=" * 60)
    
    return agent, history


def plot_training_results(episode_rewards, moving_avg_rewards, episode_losses, plot_dir):
    """Generate and save training visualization plots."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    axes[0].plot(moving_avg_rewards, label='Moving Average (100 episodes)', 
                 color='red', linewidth=2)
    axes[0].axhline(y=200, color='green', linestyle='--', 
                    label='Solved Threshold (200)', linewidth=2)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title('DQN Training Progress - LunarLander-v2', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(episode_losses, alpha=0.7, color='orange')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Average Loss', fontsize=12)
    axes[1].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    print(f"\nTraining plot saved to: {os.path.join(plot_dir, 'training_results.png')}")
    
    # Create separate reward plot for README
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)', 
             color='red', linewidth=2)
    plt.axhline(y=200, color='green', linestyle='--', 
                label='Solved Threshold (200)', linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('DQN Learning Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'reward_curve.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    agent, history = train_dqn(
        num_episodes=2000,
        max_steps=1000,
        save_dir="./models",
        plot_dir="./plots",
        log_interval=10
    )