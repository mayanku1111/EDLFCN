
import argparse
import logging
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.edlfcn import EDLFCN
from utils.data_loader import MultimodalDataset
from utils.config import load_config
from utils.metrics import calculate_metrics
from trainers.phased_trainer import PhasedTrainer
from utils.rl_utils import PPOBuffer, calculate_advantages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='EDLFCN Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint path to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    device = torch.device(config['device'])

    # Initialize components
    model = EDLFCN(config['model']).to(device)
    dataset = MultimodalDataset(config['data'])
    dataloader = DataLoader(dataset, **config['dataloader'])
    
    # Optimizers
    optimizers = {
        'pretrain': torch.optim.AdamW(
            model.pretrain_parameters(),
            lr=config['training']['pretrain_lr']
        ),
        'gan': torch.optim.AdamW(
            model.gan_parameters(),
            lr=config['training']['gan_lr']
        ),
        'main': torch.optim.AdamW(
            model.main_parameters(),
            lr=config['training']['main_lr']
        )
    }

    # Initialize trainer and buffers
    trainer = PhasedTrainer(model, optimizers, config['training']['phases'])
    replay_buffer = PPOBuffer(config['rl']['buffer_size'])
    writer = SummaryWriter(config['logging']['log_dir'])

    # Resume training if checkpoint provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Starting epoch {epoch+1}/{config['training']['epochs']}")
        phase = get_current_phase(epoch, config['training'])
        
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Train step
            step_loss = trainer.train_step(batch, phase)
            
            # RL Experience Collection
            if phase == 'main':
                with torch.no_grad():
                    outputs = model(**batch)
                    reward = calculate_reward(outputs, batch['labels'])
                    
                replay_buffer.add({
                    'states': outputs['policy_states'],
                    'actions': outputs['policy_outputs']['action'],
                    'log_probs': outputs['policy_outputs']['log_prob'],
                    'values': outputs['policy_outputs']['value'],
                    'rewards': reward
                })

            # Log metrics
            writer.add_scalar(f'Loss/{phase}', step_loss, epoch*len(dataloader)+batch_idx)

            # Periodic validation
            if (batch_idx+1) % config['training']['val_interval'] == 0:
                val_loss, val_metrics = validate(model, config['validation'])
                log_validation_metrics(writer, val_metrics, epoch*len(dataloader)+batch_idx)

            # RL Policy Update
            if phase == 'main' and len(replay_buffer) >= config['rl']['batch_size']:
                batch_data = replay_buffer.sample(config['rl']['batch_size'])
                batch_data = calculate_advantages(batch_data, config['rl']['gamma'], config['rl']['lambda'])
                policy_metrics = trainer.update_policy(batch_data)
                log_policy_metrics(writer, policy_metrics, epoch*len(dataloader)+batch_idx)
                replay_buffer.reset()

        # Save checkpoint
        if (epoch+1) % config['training']['save_interval'] == 0:
            save_checkpoint(model, optimizers, epoch, config)

    writer.close()

def get_current_phase(epoch, training_config):
    phase_thresholds = training_config['phase_thresholds']
    if epoch < phase_thresholds['pretrain']:
        return 'pretrain'
    elif epoch < phase_thresholds['gan']:
        return 'gan'
    return 'main'

def calculate_reward(outputs, labels):
    baseline_acc = (outputs['baseline_pred'].round() == labels).float().mean()
    current_acc = (outputs['prediction'].round() == labels).float().mean()
    return (current_acc - baseline_acc).clamp(min=0)

def validate(model, val_config):
    model.eval()
    val_loss = 0
    metrics = {'accuracy': 0, 'f1': 0}
    
    with torch.no_grad():
        for batch in DataLoader(MultimodalDataset(val_config['data']), 
                              batch_size=val_config['batch_size']):
            outputs = model(batch)
            val_loss += F.binary_cross_entropy(outputs['prediction'], batch['labels'])
            metrics = calculate_metrics(outputs['prediction'], batch['labels'])
    
    model.train()
    return val_loss / len(dataloader), metrics

def save_checkpoint(model, optimizers, epoch, config):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'pretrain_optimizer': optimizers['pretrain'].state_dict(),
        'gan_optimizer': optimizers['gan'].state_dict(),
        'main_optimizer': optimizers['main'].state_dict(),
        'config': config
    }
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    path = os.path.join(config['training']['save_dir'], f'checkpoint_epoch_{epoch+1}.pt')
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

if __name__ == '__main__':
    main()
