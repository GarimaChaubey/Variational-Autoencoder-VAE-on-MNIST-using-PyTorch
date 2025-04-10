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

The ELBO equation is:

![ELBO Equation](https://latex.codecogs.com/png.image?\dpi{120}&space;\log&space;p(x)&space;\geq&space;\mathbb{E}_{q(z|x)}[\log&space;p(x|z)]&space;-&space;\text{KL}(q(z|x)&space;\|&space;p(z)))

This means the VAE optimizes a combination of:
- A reconstruction loss (how well it recreates the input)
- A KL divergence loss (how close the latent distribution is to a standard Gaussian)



