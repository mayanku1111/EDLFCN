#!/usr/bin/env python3
import argparse
import logging
import torch
import numpy as np
from models.edlfcn import EDLFCN
from utils.config import load_config
from utils.data_processing import preprocess_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EDLFCNInference:
    def __init__(self, config_path, checkpoint_path):
        self.config = load_config(config_path)
        self.device = torch.device(self.config['device'])
        
        # Load model
        self.model = EDLFCN(self.config['model']).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Initialize preprocessing
        self.text_processor = preprocess_input('text')
        self.audio_processor = preprocess_input('audio')
        self.video_processor = preprocess_input('video')

    def process_inputs(self, text=None, audio=None, video=None):
        """Preprocess raw input data"""
        processed = {}
        
        if text is not None:
            processed['text'] = self.text_processor(text)
        if audio is not None:
            processed['audio'] = self.audio_processor(audio)
        if video is not None:
            processed['video'] = self.video_processor(video)
            
        return processed

    def predict(self, text=None, audio=None, video=None):
        """Run inference with optional missing modalities"""
        with torch.no_grad():
            # Preprocess inputs
            inputs = self.process_inputs(text, audio, video)
            
            # Handle missing modalities
            inputs = self._handle_missing(inputs)
            
            # Convert to tensors
            batch = {k: torch.tensor(v).unsqueeze(0).to(self.device) 
                    for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Format results
            return {
                'prediction': outputs['prediction'].squeeze().cpu().numpy(),
                'gate_scores': {k: v.squeeze().cpu().numpy() 
                              for k, v in outputs['gate_values'].items()},
                'confidence': torch.sigmoid(outputs['prediction']).item(),
                'reconstructed_modalities': {
                    k: v.squeeze().cpu().numpy() 
                    for k, v in outputs['reconstructions'].items()
                }
            }
    
    def _handle_missing(self, inputs):
        """Generate missing modalities using adversarial completion"""
        # Check missing modalities
        present_mods = set(inputs.keys())
        required_mods = {'text', 'audio', 'video'}
        
        for mod in required_mods - present_mods:
            logger.info(f"Generating missing {mod} using adversarial completion")
            inputs[mod] = self.model.adv_generator(
                context=inputs.get('text', np.random.randn(1, 256)),  # Fallback
                modality_type=mod
            )
            
        return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDLFCN Inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Input text (optional)')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to audio file (optional)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (optional)')
    
    args = parser.parse_args()
    
    # Run inference
    inferencer = EDLFCNInference(args.config, args.checkpoint)
    results = inferencer.predict(
        text=args.text,
        audio=args.audio,
        video=args.video
    )
    
    # Print results
    print("\nInference Results:")
    print(f"Predicted Sentiment: {results['prediction']:.4f}")
    print(f"Confidence: {results['confidence']:.2%}")
    print("Modality Gate Scores:")
    for mod, score in results['gate_scores'].items():
        print(f"- {mod.capitalize()}: {score:.4f}")
