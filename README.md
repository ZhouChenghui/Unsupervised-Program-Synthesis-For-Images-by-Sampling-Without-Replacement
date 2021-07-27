# Unsupervised Program Synthesis for Images using Tree-Structured LSTM 
This repository contains code accompaning the paper: [Unsupervised Program Synthesis for Images using Tree-Structured LSTM](https://arxiv.org/abs/2001.10119).

Here we only include the code for 2D CSGNet. 

### Dependency
- Python 3.*
- Please use conda env:
  ```bash
  conda env create -f requirement.txt -n CSGNet
  source activate CSGNet
  ```





### Tree LSTM RL Training
- To train a network using RL, fill up configuration in `config_tree.yml` or keep the default values and then run:
    ```
    python train_tree.py --config config_file
    ```

