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

# Acknowledgments

### Thanks to all contributors who have helped with the development of this model
