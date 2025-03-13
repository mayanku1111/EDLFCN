# Reinforcement learning policy 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RLPolicyNetwork(nn.Module):
    """
    Actor-Critic policy network for dynamic cross-modal attention weighting
    Implements proximal policy optimization (PPO) with entropy regularization
    """
    
    def __init__(self, lang_dim, audio_dim, visual_dim, hidden_dim=256):
        super().__init__()
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Shared feature encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor network (attention weights)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # Audio and visual weights
            nn.Softmax(dim=-1)
        )
        
        # Critic network (state value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, lang_feats, audio_feats, visual_feats):
        # Project features to common space
        h_lang = F.relu(self.lang_proj(lang_feats))
        h_audio = F.relu(self.audio_proj(audio_feats))
        h_visual = F.relu(self.visual_proj(visual_feats))
        
        # Concatenate and encode
        combined = torch.cat([h_lang, h_audio, h_visual], dim=-1)
        state = self.shared_encoder(combined)
        
        # Get action probabilities and state value
        action_probs = self.actor(state)
        state_value = self.critic(state)
        
        return action_probs, state_value
    
    def get_action(self, lang_feats, audio_feats, visual_feats):
        # Get probability distribution
        probs, value = self(lang_feats, audio_feats, visual_feats)
        distribution = Categorical(probs)
        
        # Sample action
        action = distribution.sample()
        
        # Calculate log probability
        log_prob = distribution.log_prob(action)
        
        return {
            'action': action,
            'log_prob': log_prob,
            'value': value.squeeze(-1),
            'entropy': distribution.entropy()
        }
    
    def evaluate_actions(self, lang_feats, audio_feats, visual_feats, actions):
        # Get current probabilities
        probs, value = self(lang_feats, audio_feats, visual_feats)
        distribution = Categorical(probs)
        
        # Calculate metrics
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return log_prob, entropy, value.squeeze(-1)

class RLPolicyManager:
    """
    Manages policy updates using PPO-clip objective
    """
    
    def __init__(self, policy_net, optimizer, clip_epsilon=0.2, entropy_coef=0.01):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
    def update_policy(self, batch_data):
        # Unpack batch data
        states = {k: v.detach() for k, v in batch_data.items()}
        old_log_probs = states['log_probs'].detach()
        actions = states['actions'].detach()
        returns = states['returns'].detach()
        advantages = states['advantages'].detach()
        
        # Calculate new policy metrics
        new_log_probs, entropy, values = self.policy_net.evaluate_actions(
            states['lang_feats'],
            states['audio_feats'],
            states['visual_feats'],
            actions
        )
        
        # Calculate policy ratio
        ratio = (new_log_probs - old_log_probs).exp()
        
        # PPO-clip objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy regularization
        entropy_loss = -entropy.mean() * self.entropy_coef
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }
