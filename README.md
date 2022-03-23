# Unsupervised Program Synthesis for Images By Sampling Without Replacement
This repository contains code accompaning the paper: [Unsupervised Program Synthesis for Images By Sampling Without Replacement](https://proceedings.mlr.press/v161/zhou21b.html).

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
- Before training, please create the log folders. There should be:
    ```
    log\logger
    log\tensorboard
    log\configs
    ```
- To train a network using RL, fill up configuration in `config/config_tree_cad.yml` or keep the default values and then run:
    ```
    python train_tree.py --config config_tree_cad.yml
    ```
    Note that you should adjust the `beam_n` and `batch_size` parameters to fit your GPU. 

### File Pointers

- The implementation of the SWOR tree LSTM can be found in `src/Models/tbs_model.py`.
- The implementation of the SWOR RL objective can be found in `src/utils/reinforce_tree.py`.
