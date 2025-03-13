class LCCAAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lang_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mod_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_net = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # RL Components
        self.policy_layer = nn.Linear(hidden_dim, 1)
        self.reward_buffer = []

    def forward(self, lang_feats, mod_feats):
        # Project features
        Q = self.lang_proj(lang_feats)
        K = self.mod_proj(mod_feats)
        V = mod_feats
        
        # Compute attention weights with RL guidance
        attn_weights, _ = self.attention_net(Q, K, V)
        policy_weights = torch.sigmoid(self.policy_layer(attn_weights))
        
        # Store for reward calculation
        if self.training:
            self.reward_buffer.append(policy_weights.detach())
            
        return policy_weights * attn_weights
