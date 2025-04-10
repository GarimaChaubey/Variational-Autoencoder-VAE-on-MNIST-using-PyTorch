# Variational-Autoencoder-VAE-on-MNIST-using-PyTorch

This repository contains an implementation of a Variational Autoencoder (VAE) using PyTorch on the MNIST dataset. The goal is to learn a compressed latent representation of digit images and reconstruct them while understanding the generative capabilities of VAEs.

## What is a VAE?
A Variational Autoencoder (VAE) is a type of generative model that learns not just to compress and reconstruct data, but also to generate new data samples by learning a latent distribution. It builds on top of the traditional autoencoder by incorporating principles from Bayesian inference.

![image](https://github.com/user-attachments/assets/903fbf45-a1ab-4a93-8854-aaf6993378d2)

## From Autoencoder to VAE

| Autoencoder                                   | Variational Autoencoder                                          |
|----------------------------------------------|------------------------------------------------------------------|
| Learns deterministic latent vectors          | Learns a **probability distribution** over the latent space      |
| Encodes directly to a point                  | Encodes to **mean (μ)** and **log variance (log σ²)**            |
| Decoder maps fixed code to reconstruction    | Decoder samples from learned distribution using **reparameterization** |
| Not generative                               | Fully **generative**                                             |

## Mathematical Intuition

The Variational Autoencoder (VAE) aims to maximize the likelihood of input data `p(x)` by optimizing the **Evidence Lower Bound (ELBO)**:

**ELBO Objective:**

    log p(x) ≥ E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

### 🔍 Explanation:

- **Reconstruction Loss** `E_q(z|x)[log p(x|z)]`:  
  Encourages the decoder to reconstruct the input accurately from the latent vector.

- **KL Divergence** `KL(q(z|x) || p(z))`:  
  A regularization term that aligns the learned latent distribution `q(z|x)` with a prior distribution `p(z) ~ N(0, I)`.

This balance ensures the VAE learns a compressed latent space while keeping it smooth and continuous, ideal for generating new data.

## Architecture

### Encoder

The encoder compresses the **28x28** input image into a latent space using linear layers and computes:

- Mean vector: `μ`
- Log-variance vector: `log(σ²)`

These are the learned parameters representing the distribution of the latent space.

---

### Reparameterization Trick

To allow gradients to flow through the stochastic latent space, we apply the reparameterization trick:

`z = μ + σ * ε`, where `ε ∼ N(0, I)`

This enables sampling from the distribution while maintaining differentiability for backpropagation.

---

### Decoder

The decoder reconstructs the original input from the sampled latent vector `z`.

It expands the latent representation back to a `28x28` image using fully connected layers.

## Result 
### Training Loss

![image](https://github.com/user-attachments/assets/98e88b1e-ec1d-476f-905a-ecfdde63437d)

The graph above shows the training loss curve for the Variational Autoencoder (VAE) over 10 epochs.

- At epoch 0, the loss is quite high (~103), which is expected as the model starts with random weights.

- As the training progresses, the loss consistently decreases, indicating that the model is successfully learning to encode and decode MNIST digits.

- By epoch 9, the loss stabilizes around ~49, suggesting convergence and effective learning of the latent space representation.

This declining trend validates that the combination of reconstruction loss and KL divergence is being minimized effectively, leading to meaningful latent representations.

### Original (Top) vs Reconstructed (Bottom)
![image](https://github.com/user-attachments/assets/668dcae0-73ca-4b6f-a21f-90a676d42408)

The image above compares the original MNIST digits (top row) with their reconstructions by the VAE (bottom row) after training.

- We can see that the reconstructed digits retain the overall shape and identity of the original images.

- Although there's a slight blurriness in the reconstructions (a known trait of VAEs due to their probabilistic nature), the model has successfully captured the semantic structure of the digits.

- This reflects that the encoder has learned a meaningful latent representation, and the decoder is able to reconstruct from it with decent accuracy.

This qualitative evaluation complements the decreasing training loss, confirming that the VAE is functioning effectively.

## Refrences
-https://arxiv.org/pdf/1312.6114
