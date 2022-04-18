# oct-for-spoofing-detection
This is the offical implementation of our SPL paper "The Role of Long-term Dependency in Synthetic Speech Detection".

# Requirements
Python, Spafe, Librosa, Numpy, Pytorch and Tensorflow (only used for calculating supervised contrastive loss)

# Data Preparation
The LFCC features used in this implementation is either extracted by the Matlab code provided by ASVspoof organizer in the competition baseline, or extracted based on the Spafe library. Our experiments are based on the **Matlab** code. However, the method based on Spafe library is easier to use in practice.

# Run the training code

