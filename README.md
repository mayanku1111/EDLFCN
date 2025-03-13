# EDLFCN

## This repo is about sentiment analysis using Enhanced Disentangled-Language-Focused Collaborative Network

## This is a robust framework for multimodal sentiment analysis that effectively handles noisy and incongruent data in real-world scenarios.

![Image](https://github.com/user-attachments/assets/531ab630-5fbd-4c31-9ace-ea058d828ced)

# Architecture Overview
### EDLFCN builds upon the DLF (Disentangled-Language-Focused) architecture with six novel components designed to enhance performance and robustness:

## 1. Dynamic Modality Gating
Adaptively weights modalities based on quality and alignment with language:

Computes gating scores: gm = σ(Wg[Entropy(Xm); Sim(Xm, XL)])
Emphasizes cleaner modalities in shared space while preserving unique features

## 2. Cross-Modal Consistency
Applies contrastive learning to align modalities:

Alignment loss: Lalign = -log(e^s(Shm,Shn)/τ / ∑e^s(Shm,Shk)/τ)
Preserves relationships between different modalities' shared features

## 3. Language-Guided Attention (LCCA)
Reinforcement learning optimizes attention weights:

Attention weights: α = softmax(QlangK⊤mod)
Policy gradient update: ∇J ∝ R · ∇ log π(α|X)
Dynamically focuses on the most relevant audio/visual features

## 4. Multi-Scale Fusion
Implements hierarchical processing across multiple linguistic levels:

Ffinal = [CNNword(HL); BiGRUphrase(HL); Transformerutterance(HL)]

## 5. Adversarial Completion
GAN with language guidance for robust missing data reconstruction:

Generator: Xm_rec = G(Xm_noisy, HL)
Discriminator detects real/fake features
Loss: LGAN + λ||Xm - Xm_rec||1

## 6. Cross-Modal Orthogonal Projection
Ensures separation between shared and specific subspaces:

Minimizes redundancy across modalities
Enforces orthogonality constraints

# Acknowledgments

### Thanks to all contributors who have helped with the development of this model
