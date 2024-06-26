# New boot goofin with PyTorch

Playing around with PyTorch as I review, explore, and re-explore Deep Learning basics,
practices, and implementation strategies.

## Contents

```
pytorch_fun
 ┣ data  -----------------------> Data lives here locally
 ┃ ┗ .gitkeep
 ┣ docs
 ┃ ┗ resources.md  -------------> Some resources I've used
 ┣ models
 ┃ ┣ early_adversarial_droput --> Early stopping, adversarial training, and dropout
 ┃ ┃ ┣ adversarial.py  
 ┃ ┃ ┗ training.ipynb
 ┃ ┣ weight_decay --------------> L1/L2 weight decay regularization
 ┃ ┃ ┣ nets.py  
 ┃ ┃ ┗ training.ipynb
 ┃ ┗ xor  ----------------------> XOR implementation with 2D MLP
 ┃ ┃ ┣ net.py  
 ┃ ┃ ┗ training.ipynb
 ┣ .gitignore
 ┣ mixin.py  -------------------> Mixin nn.Module for basic boiler plate
 ┗ README.md
 ```