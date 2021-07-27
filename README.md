# Unsupervised Program Synthesis for Images using Tree-Structured LSTM 
This repository contains code accompaning the paper: [Unsupervised Program Synthesis for Images using Tree-Structured LSTM](https://arxiv.org/abs/2001.10119).

We include code that learns to approximate CAD chair dataset without supervision.  

### Dependency
- Python 3.*
- Please use conda env:
  ```bash
  conda env create -f requirement.txt -n CSGNet
  source activate CSGNet
  ```

###Data

- CAD furniture dataset is included in the `data/cad` directory.  



### RL Training
- To train a network using RL, fill up configuration in `config/config_tree_cad.yml` or keep the default values and then run:
    ```
    python train_tree.py --config config_file
    ```

