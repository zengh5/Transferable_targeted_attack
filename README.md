# Transferable_targeted_attack
PyTorch code for our submitted paper:

**"Enhancing Targeted Transferability via suppressing high-confidence labels"**. H. Zeng, K. Yu, B. Chen, and A. Peng.

### Motiviation
Let's say we attacked a 'tench' image to a 'otterhound', and succeed in the source model. When the attacked image is transferred to a target model, the most likely output is: 'otterhound' (great!), 'tench' (oops!), other fish-like labels (oops!), and dog-like labels (oops!). 1) the original label is likely to be 'restored' in the target model; 2) the adversarial perturbation may be explained as dog-like, rather than the exact 'otterhound' in the target model.  
To this end, we propose a new Transferable Targeted attack as following:
<p align="left">
  <img src="https://github.com/zengh5/Transferable_targeted_attack/tree/blob/Figures/Fig1_target_transfer.png" width='700'>
</p>

### Dataset
The 1000 images from the NIPS 2017 ImageNet-Compatible dataset. [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) or [Zhao's Github](https://github.com/ZhengyuZhao/Targeted-Tansfer/tree/main/dataset). 

### Evaluation
We compare the proposed method with three simple transferable targeted attacks (CE, Po+Trip, and Logit).
All attacks are integrated with TI, MI, and DI, and run with 200 iterations to ensure convergence.
L<sub>&infin;</sub>=16 is applied.

#### ```eval_single.py```: Single-model transfer.
#### ```eval_ensemble.py```: Ensemble transfer. 

Our codes are heavily borrowed from:
https://github.com/ZhengyuZhao/Targeted-Tansfer
