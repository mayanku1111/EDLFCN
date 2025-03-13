class OrthogonalLoss(nn.Module):
    def __init__(self, lambda_ortho=0.1):
        super().__init__()
        self.lambda_ortho = lambda_ortho

    def forward(self, shared_feats, specific_feats):
        batch_size = shared_feats.size(0)
        loss = 0
        
        for i in range(batch_size):
            s = shared_feats[i]
            p = specific_feats[i]
            ortho = torch.norm(tch.mm(s.t(), p)) / (torch.norm(s) * torch.norm(p))
            loss += self.lambda_ortho * ortho
            
        return loss / batch_size
