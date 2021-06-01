import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        #nn.init.xavier_uniform(self.fc3.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.checkpoint_filepath = os.path.join('', "pg_net")
    
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_filepath)

    def load_checkpoint(self, file):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(file))
    

class PG_Agent(object):
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4,
                l1_size=256, l2_size=256):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(lr, input_dims, l1_size, l2_size, n_actions)
    
    def choose_action(self, observation):
        probabilities = F.softmax(self.policy.forward(observation))
        # That's a bit confusing.
        # Basically we sample an action (do we need to use Categorical?),
        # and then compute the logarithm of probablility of taking that action
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)
    
    def save_model(self):
        self.policy.save_checkpoint()
    
    def load_model(self, filepath):
        self.policy.load_checkpoint(filepath)

    def learn(self):
        T.cuda.empty_cache()
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        # wait, isn't there a simpler method to do this?
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
            
        # Normalize
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std

        # We send to device because it needs to be a cuda tensor apparently
        G = T.tensor(G, dtype=T.float).to(self.policy.device) 

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []