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

The Variational Autoencoder (VAE) seeks to **maximize the likelihood** of the observed data \( p(x) \) indirectly by optimizing the **Evidence Lower Bound (ELBO)**:

\[
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))
\]

### 🔍 Explanation:

- **First Term** \( \mathbb{E}_{q(z|x)}[\log p(x|z)] \):  
  This is the **reconstruction loss**, which encourages the decoder to produce outputs close to the input data. It's typically calculated using Binary Cross-Entropy (BCE) or Mean Squared Error (MSE).

- **Second Term** \( \text{KL}(q(z|x) \| p(z)) \):  
  This is the **Kullback–Leibler divergence**, which regularizes the latent space by forcing the approximate posterior \( q(z|x) \) to be close to the prior distribution \( p(z) \sim \mathcal{N}(0, I) \).

> This balance allows VAEs to learn structured, continuous, and meaningful latent representations that can be sampled from during generation.


