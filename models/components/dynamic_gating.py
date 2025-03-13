class DynamicModalityGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.entropy_layer = nn.Linear(hidden_dim, 1)
        self.similarity_layer = nn.Linear(hidden_dim * 2, 1)
        self.gate_network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def compute_entropy(self, x):
        return -torch.sum(x * torch.log(x + 1e-8), dim=-1)

    def forward(self, modality_feats, lang_feats):
        # Calculate entropy and similarity
        entropy = self.compute_entropy(modality_feats)
        similarity = F.cosine_similarity(modality_feats, lang_feats, dim=-1)
        
        # Combine features for gating
        gate_input = torch.stack([entropy, similarity], dim=-1)
        return self.gate_network(gate_input)
