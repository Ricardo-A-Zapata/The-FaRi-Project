# The FaRi Project - Voice Replication using CycleGAN

## Overview

The **FaRi Project** is a deep learning initiative developed by Faith Villarreal and Ricardo Zapata. The project focuses on replicating voices using a **Cycle Generative Adversarial Network (CycleGAN)** model. This model enables users to input a recording of their own voice and replicate the voice of prominent figures or musical artists through deep learning techniques.

In this project, a CycleGAN is leveraged to transform one domain (the user’s voice) into another (the voice of a target artist or figure). The process ensures that after transforming the input, the output retains a similar structure to the original, yet mimics the voice characteristics of the target.

## What is a CycleGAN?

A **Cycle Generative Adversarial Network (CycleGAN)** is a deep learning model that extends the classic GAN (Generative Adversarial Network) architecture. A traditional GAN consists of two neural networks:
- A **Generator**, which creates data.
- A **Discriminator**, which evaluates and attempts to distinguish between real and generated data.

These two models engage in an adversarial process to improve each other. The **CycleGAN** modifies this by introducing two GANs to map data between two domains (e.g., user voice to target voice and vice versa). The model also incorporates a **cycle consistency loss**, ensuring that if the data is transformed from one domain to another and back, the result remains consistent with the original.

## Project Goal

The primary goal of this project is to train models using voice data from cultural and societal figures. Users can then input a recording of their own voice, and the model will generate an output that mimics the voice of the chosen figure. This could have exciting implications for media, entertainment, and personalization in vocal recordings.

## Project Structure

- **Data Preprocessing:** Prepares and cleans the voice data for training.
- **CycleGAN Architecture:** Implements the CycleGAN model for voice transformation.
- **Training Process:** Trains the CycleGAN on voice datasets.
- **Griffin-Lim Vocoder:** Converts spectrograms back into waveform audio.
- **Data Loader:** Handles loading and batching of the voice datasets for training.

## Technologies and Libraries

- Python
- TensorFlow / PyTorch (deep learning frameworks)
- Jupyter Notebook
- Griffin-Lim Vocoder

## Installation and Usage

### Prerequisites:
- Install the necessary libraries listed in `requirements.txt` (if applicable).
- Python 3.x environment with Jupyter Notebook support.

### Running the Project:
1. Clone the repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/The-FaRi-Project.git
