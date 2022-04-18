# OCT-for-spoofing-detection
This is the offical implementation of our SPL paper "The Role of Long-term Dependency in Synthetic Speech Detection". We also provide the raw data used to generate the results in our paper. The pretrained checkpoint can be found in the checkpoint directory, and the countermeasure scores are included in ./tDCF_python_v2/scores. 

# Requirements
Python, Spafe, Librosa, Numpy, Pytorch and Tensorflow (only used for calculating supervised contrastive loss)

# Data Preparation
The LFCC features used in this implementation are either extracted by the Matlab code provided by ASVspoof organizers in the competition baseline, or extracted with the Spafe library. Our experiments are based on the **Matlab** code. However, feature extraction based on Spafe library is easier to use in practice.

# How to use

1. To get Fig.1 in our paper, go to the notebook named SCL_analysis. There are detailed instructions on how to produce this figure. The raw data used to generate Fig.1 in our paper is also included in this notebook.
2. To train you own model, simply run
   ```python train.py --finetune --epochs 300```
3. To do inference with our/your checkpoint, see the notebook named inference. After finishing inference and getting the score file, you can use tDCF_python_v2 directory which is provided by ASVspoof organizers to get EER and t-DCF.

# Citation
This part will be updated after uploading final files.

