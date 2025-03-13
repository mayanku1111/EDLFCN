class CrossModalConsistency(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)

    def forward(self, shared_features):
        # shared_features: list of [sh_a, sh_v, sh_l]
        losses = []
        for i in range(len(shared_features)):
            anchor = shared_features[i]
            positives = [f for j, f in enumerate(shared_features) if j != i]
            
            pos_sim = torch.exp(self.cossim(anchor, positives[0]) / self.temp)
            neg_sim = torch.sum(torch.exp(
                torch.stack([self.cossim(anchor, neg) / self.temp for neg in positives[1:]])
            ), dim=0)
            
            losses.append(-torch.log(pos_sim / (pos_sim + neg_sim)))
            
        return torch.mean(torch.stack(losses))
