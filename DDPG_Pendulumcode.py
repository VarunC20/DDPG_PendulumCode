import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import pygame
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Tanh for [-1, 1] range
        return x * self.action_bound  # Scale to action space bounds

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, use_pretrained, model_dir = "models"):
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.critic_loss = nn.MSELoss()

        self.replay_buffer = collections.deque(maxlen=100000)
        self.discount_factor = 0.99
        self.tau = 0.01  # Soft update rate
        self.action_bound = action_bound.item()
        self.model_dir = model_dir

        if use_pretrained:
            self.load_model()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        noise = np.random.normal(0, 0.2, size=action.shape)
        action = np.clip(action+noise, -1, 1)
        return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch)) #zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.discount_factor * self.target_critic(next_states, next_actions) * (1 - dones.int())
        current_q = self.critic(states, actions)
        critic_loss = self.critic_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(self.model_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(self.model_dir, "critic.pth"))
        print("Model saved successfully.")

    def load_model(self):
        actor_path = os.path.join(self.model_dir, "actor.pth")
        critic_path = os.path.join(self.model_dir, "critic.pth")

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=device))
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            print("Pre-trained model loaded successfully.")
        else:
            print("No pre-trained model found. Training from scratch.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Pendulum-v1", render_mode="rgb_array")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)

use_pretrained = input("Use pre-trained model? (y/n): ").strip().lower() == "y"
agent = DDPGAgent(state_dim, action_dim, action_bound, use_pretrained)

episodes = 1000
batch_size = 256
reward_trends = []

width, height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    act = []

    for _ in range(200):  # Limit the number of steps per episode
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.append((state, action, reward, next_state, done))
        agent.update(batch_size)

        state = next_state
        act.append(action)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((0, 0, 0))  
        img = np.transpose(env.render(), (1, 0, 2))  
        img = pygame.surfarray.make_surface(img)  
        screen.blit(img, (0, 0))  
        pygame.display.update()  

        clock.tick(30)  # Control the frame rate

        if done:
            break

    reward_trends.append(total_reward)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}") 

agent.save_model()
pygame.quit()

# Plot reward trends
plt.plot(reward_trends)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Trend Over Episodes")
plt.show()
