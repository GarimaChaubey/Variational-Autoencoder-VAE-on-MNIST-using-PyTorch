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

## 📐 Mathematical Intuition

The Variational Autoencoder (VAE) aims to maximize the likelihood of input data `p(x)` by optimizing the **Evidence Lower Bound (ELBO)**:

**ELBO Objective:**

    log p(x) ≥ E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

### 🔍 Explanation:

- **Reconstruction Loss** `E_q(z|x)[log p(x|z)]`:  
  Encourages the decoder to reconstruct the input accurately from the latent vector.

- **KL Divergence** `KL(q(z|x) || p(z))`:  
  A regularization term that aligns the learned latent distribution `q(z|x)` with a prior distribution `p(z) ~ N(0, I)`.

This balance ensures the VAE learns a compressed latent space while keeping it smooth and continuous, ideal for generating new data.




