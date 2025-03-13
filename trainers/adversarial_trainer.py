class AdversarialTrainer:
    def __init__(self, generator, discriminator, g_optim, d_optim):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optim = g_optim
        self.d_optim = d_optim
        
    def train_step(self, real_data, lang_context):
        # Generate fake data
        fake_data = self.generator(real_data, lang_context)
        
        # Train Discriminator
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())
        
        d_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)) + \
                 F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
        
        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()
        
        # Train Generator
        fake_pred = self.discriminator(fake_data)
        recon_loss = F.l1_loss(fake_data, real_data)
        g_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred)) + \
                 0.5 * recon_loss
                 
        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        
        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}
