# evaluate.py
import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
import argparse
import time

def evaluate_agent(model_path, num_episodes=100, render=False, record_video=False):
    """
    Evaluate a trained DQN agent.
    
    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        record_video: Whether to record evaluation videos
    """
    # Create environment
    if record_video:
        env = gym.make('LunarLander-v2', render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, './videos', episode_trigger=lambda x: True)
    elif render:
        env = gym.make('LunarLander-v2', render_mode='human')
    else:
        env = gym.make('LunarLander-v2')
    
    # Initialize and load agent
    agent = DQNAgent()
    agent.load(model_path)
    print(f"Loaded model from: {model_path}")
    print(f"Evaluation epsilon: {agent.epsilon}")
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0  # Episodes with reward > 200
    
    print("\n" + "=" * 60)
    print(f"Evaluating agent for {num_episodes} episodes")
    print("=" * 60)
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if render and not record_video:
                time.sleep(0.01)  # Slow down for visualization
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode_reward >= 200:
            success_count += 1
        
        print(f"Episode {episode}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Steps: {steps}")
    
    env.close()
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = (success_count / num_episodes) * 100
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Episode Length: {mean_length:.2f} steps")
    print(f"Success Rate (≥200): {success_rate:.1f}% ({success_count}/{num_episodes})")
    print("=" * 60)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards
    }


def test_single_episode(model_path, render=True):
    """Run a single episode with visualization."""
    print("Running single test episode...")
    return evaluate_agent(model_path, num_episodes=1, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DQN agent on LunarLander')
    parser.add_argument('--model', type=str, default='./models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--record', action='store_true',
                        help='Record evaluation videos')
    parser.add_argument('--test', action='store_true',
                        help='Run single test episode with rendering')
    
    args = parser.parse_args()
    
    if args.test:
        test_single_episode(args.model, render=True)
    else:
        evaluate_agent(args.model, args.episodes, args.render, args.record)