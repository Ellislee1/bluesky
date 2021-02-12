# import numba as nb
import numpy as np
import torch
import torch.nn as nn
# Import the global bluesky objects. Uncomment the ones you need
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, action_std):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.action_var = torch.full(
            (action_size,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_size, action_size, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_size, action_size, action_std).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr)

        self.policy_old = ActorCritic(
            state_size, action_size, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.policy.eval()

    def save_model(self, PATH):
        torch.save(self.policy.state_dict(), PATH)

    def load_model(self, PATH):
        self.policy.load_state_dict(torch.load(PATH))
        self.policy_old.load_state_dict(torch.load(PATH))

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(
            state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal == 1:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-5)).float()

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(
            memory.states), 1).to(device).float().detach()
        old_actions = torch.squeeze(torch.stack(
            memory.actions), 1).to(device).float().detach()
        old_logprobs = torch.squeeze(torch.stack(
            memory.logprobs), 1).to(device).float().detach()

        # Optimize policy for K epochs:
        for _ in range(5):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach()).float()

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -(-torch.min(surr1, surr2) + 0.5 *
                     self.MseLoss(state_values, rewards) - 0.01*dist_entropy)

            loss = loss.float()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
