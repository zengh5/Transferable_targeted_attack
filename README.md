# Transferable_targeted_attack
PyTorch code for our accepted paper by 2023 ICIP:

**"Enhancing Targeted Transferability via suppressing high-confidence labels"**. H. Zeng, T. Zhang, B. Chen, and A. Peng.

### Motiviation
Let's say we attacked a 'tench' image to an 'otterhound' one, and succeed in the source model. When the attacked image is transferred to a target model, the most likely outputs are: 'otterhound' (great!), 'tench' (oops!), other fish-like labels (oops!), and dog-like labels (oops!). This result implies that:  
- The original label is likely to be 'restored' in the target model;  

- High-confidence labels in the source model are likely to retain high confidence in the target model, no matter for original images (below left) or attacked ones (below right). This means the adversarial perturbation may be explained as dog-like, rather than the exact 'otterhound' in the target model. In the bottom left, the high/low-confident labels are calculated on original images. The original label is excluded. In the bottom right, the high/low-confident labels are calculated on attacked images (targeted TMDI, eps=16). both the original label and the target label are excluded. 
<p align="center">
  <img src="https://github.com/zengh5/Transferable_targeted_attack/blob/main/Figures/highlow_conf.png" width='400'>
  <img src="https://github.com/zengh5/Transferable_targeted_attack/blob/main/Figures/highlow_conf_AE.png" width='400'>
</p>

To this end, we propose a new Transferable Targeted attack as following:
<p align="center">
  <img src="https://github.com/zengh5/Transferable_targeted_attack/blob/main/Figures/Fig1_target_transfer.png" width='500'>
</p>

### Dataset
The 1000 images are from the NIPS 2017 ImageNet-Compatible dataset. [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) or [Zhao's Github](https://github.com/ZhengyuZhao/Targeted-Tansfer/tree/main/dataset). 

### Evaluation
We compare the proposed method with three simple transferable targeted attacks (CE, Po+Trip, and Logit).
All attacks are integrated with TI, MI, and DI, and run with 200 iterations to ensure convergence.
L<sub>&infin;</sub>=16 is applied.

#### ```eval_single_TMDI.py```: Single-model transfer.
#### ```eval_ensemble_TMDI.py```: Ensemble transfer. 

### More results
We provide ablation study on N_h and T, visual quality comparison, and attack ability comparison with a generative attack TTP, in the 'supp.pdf'.

Our codes are heavily borrowed from:
https://github.com/ZhengyuZhao/Targeted-Transfer
