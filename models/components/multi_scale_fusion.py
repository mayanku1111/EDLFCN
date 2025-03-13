import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusion(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super().__init__()
        # Word-level CNN
        self.word_conv = nn.Sequential(
            nn.Conv1d(text_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        
        # Phrase-level BiGRU
        self.phrase_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True)
        
        # Utterance-level Transformer
        self.utterance_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim*2, nhead=4),
            num_layers=2)
        
        # Fusion layer
        self.fusion_proj = nn.Linear(hidden_dim*4, hidden_dim)

    def forward(self, text_features):
        # Word-level features
        word_feats = self.word_conv(text_features.permute(0,2,1)).squeeze(-1)
        
        # Phrase-level features
        phrase_feats, _ = self.phrase_gru(text_features)
        phrase_feats = phrase_feats.mean(dim=1)
        
        # Utterance-level features
        utterance_feats = self.utterance_transformer(text_features)
        utterance_feats = utterance_feats.mean(dim=1)
        
        # Multi-scale fusion
        combined = torch.cat([
            word_feats, 
            phrase_feats, 
            utterance_feats,
            text_features.mean(dim=1)
        ], dim=-1)
        
        return self.fusion_proj(combined)
