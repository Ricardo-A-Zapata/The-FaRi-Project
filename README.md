# The FaRi Project - Voice Replication using CycleGAN

### Overview

The **FaRi Project** is a deep learning initiative developed by Faith Villarreal and Ricardo Zapata. The project focuses on replicating voices using a **Cycle Generative Adversarial Network (CycleGAN)** model. This model enables users to input a recording of their own voice and replicate the voice of prominent figures or musical artists through deep learning techniques.

In this project, a CycleGAN is leveraged to transform one domain (the user’s voice) into another (the voice of a target artist or figure). The process ensures that after transforming the input, the output retains a similar structure to the original, yet mimics the voice characteristics of the target.

## What is a CycleGAN?

A **Cycle Generative Adversarial Network (CycleGAN)** is a deep learning model that extends the classic GAN (Generative Adversarial Network) architecture. A traditional GAN consists of two neural networks:
- A **Generator**, which creates data.
- A **Discriminator**, which evaluates and attempts to distinguish between real and generated data.

These two models engage in an adversarial process to improve each other. The **CycleGAN** modifies this by introducing two GANs to map data between two domains (e.g., user voice to target voice and vice versa). The model also incorporates a **cycle consistency loss**, ensuring that if the data is transformed from one domain to another and back, the result remains consistent with the original.

## Project Goal

The primary goal of this project is democratize music creation and free expression. People should not be limited by their finances or ability to access professional studio/mix engineering to express themselves. Through The FaRi Project, we hope to help anyone and everyone have access to professional - sounding music in a easy-to-use way. We hope to allow users to provide stems such as raw vocal recordings and instrumentation / a beat, using our trained model to generate a fully mixed sound that is output at the end. We also plan to implement an AI ChatBot that will ask the user what they think of the final mix and be receptive to feedback in order to fix the mix to the user's liking. This could have exciting implications for media, entertainment, and personalization in vocal recordings.

## Project Structure

- **Data Preprocessing:** Prepares and cleans the voice data for training.
- **CycleGAN Architecture:** Implements the CycleGAN model for voice transformation.
- **Training Process:** Trains the CycleGAN on voice datasets.
- **Griffin-Lim Vocoder:** Converts spectrograms back into waveform audio.
- **Data Loader:** Handles loading and batching of the voice datasets for training.

## Technologies and Libraries

- Python
- TensorFlow / PyTorch (deep learning frameworks)
- Griffin-Lim Vocoder

## Installation and Usage

### Prerequisites:
- Install the necessary libraries listed in `requirements.txt`.

# Credits and Thanks

## In search of resources to learn more about deep learning, adversarial networks, and generative adversarial networks for this project, the following videos/resources were very helpful so I want to show appreciation and catalog them below:

[Deep Learning Crash Course for Beginners]( https://www.youtube.com/watch?v=VyWAvY2CF9c&t=3387s&ab_channel=freeCodeCamp.org) by freeCodeCamp.org

[Generative Adversarial Networks (GANs) - Computerphile](https://www.youtube.com/watch?v=Sw9r8CL98N0&ab_channel=Computerphile) by Computerphile

[Zebras, Horses & CycleGAN - Computerphile](https://www.youtube.com/watch?v=T-lBMrjZ3_0&ab_channel=Computerphile) by Computerphile

[Deep Learning 46: Unpaired Image to Image translation Network (Cycle GAN) and DiscoGAN](https://www.youtube.com/watch?v=nB8uVGbesZ4&ab_channel=AhladKumar) by Ahlad Kumar

[Deep Learning 47: TensorFlow Implementation of Image to Image Translation Network (Cycle GAN)](https://www.youtube.com/watch?v=nwtWt4tTm9s&ab_channel=AhladKumar) by Ahlad Kumar

[AI Audio Datasets (AI-ADS) GitHub](https://github.com/Yuan-ManX/ai-audio-datasets) from GitHub user Yuan ManX

[CycleGAN GitHub](https://github.com/junyanz/CycleGAN) from GitHub user junyanz
