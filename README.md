# OCT-for-spoofing-detection
This is the offical implementation of our SPL paper "The Role of Long-term Dependency in Synthetic Speech Detection".

# Requirements
Python, Spafe, Librosa, Numpy, Pytorch and Tensorflow (only used for calculating supervised contrastive loss)

# Data Preparation
The LFCC features used in this implementation is either extracted by the Matlab code provided by ASVspoof organizers in the competition baseline, or extracted based on the Spafe library. Our experiments are based on the **Matlab** code. However, the method based on Spafe library is easier to use in practice.

# How to use

1. To get Fig.1 in our paper, go to the notebook named SCL_analysis. There are detailed instruction on how to produce this figure. The raw data used to generate Fig.1 is also included in this notebook.
2. To train you own model, run
   ```python train.py --finetune --epochs 300```
3. To do inference with our/your checkpoint, see the notebook named inference. Once after finishing inference and get the score file, you can use tDCF_python_v2 directory which is provided by ASVspoof organizers to get EER and t-DCF.

# Citation
This part will be updated after online-pub.

