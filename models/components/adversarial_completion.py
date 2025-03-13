class AdversarialGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, args.hidden_dim)
        )
        
    def forward(self, noisy_input, lang_context):
        combined = torch.cat([noisy_input, lang_context], dim=-1)
        latent = self.encoder(combined)
        return self.decoder(latent)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
