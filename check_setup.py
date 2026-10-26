import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0, seed=0)
model.learn(total_timesteps=5000)
print("Setup OK")