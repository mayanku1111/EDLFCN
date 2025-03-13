class PhasedTrainer:
    def __init__(self, model, optimizers, phases):
        self.model = model
        self.optimizers = optimizers  # Dict with 'disentangle', 'gan', 'main'
        self.phases = phases
        
    def train_step(self, batch, phase):
        # Set model modes
        self.model.train()
        if phase == 'pretrain':
            self.model.adv_generator.eval()
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Phase-specific loss calculation
        loss = 0
        if phase == 'pretrain':
            loss += self._disentanglement_loss(outputs)
            loss += self._alignment_loss(outputs)
        elif phase == 'gan':
            loss += self._adversarial_loss(outputs)
        else:
            loss += self._main_loss(outputs)
            loss += self._rl_loss(outputs)
        
        # Backpropagation
        self.optimizers[phase].zero_grad()
        loss.backward()
        self.optimizers[phase].step()
        
        return loss.item()

    def _disentanglement_loss(self, outputs):
        # Reconstruction loss
        rec_loss = F.mse_loss(outputs['recon_a'], outputs['origin_a']) + \
                   F.mse_loss(outputs['recon_v'], outputs['origin_v'])
        
        # Orthogonality constraint
        ortho_loss = torch.mean(
            torch.abs(torch.sum(outputs['s_a'] * outputs['c_a'], dim=-1)) + \
            torch.mean(torch.abs(torch.sum(outputs['s_v'] * outputs['c_v'], dim=-1))
            
        return rec_loss + 0.1 * ortho_loss

    def _adversarial_loss(self, outputs):
        # Generator loss
        real_pred = self.model.discriminator(outputs['origin_a'])
        fake_pred = self.model.discriminator(outputs['recon_a'])
        gen_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
        
        # Discriminator loss
        disc_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)) + \
                    F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
        
        return gen_loss + disc_loss

    def _rl_loss(self, outputs):
        # Calculate policy gradient loss
        advantages = outputs['value_estimate'].detach() - outputs['rewards']
        policy_loss = -torch.mean(outputs['action_probs'] * advantages)
        
        # Value loss
        value_loss = F.mse_loss(outputs['value_estimate'], outputs['rewards'])
        
        return policy_loss + 0.5 * value_loss
