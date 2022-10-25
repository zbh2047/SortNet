# SortNet

## Introduction

This is the official repo for training SortNet, a novel 1-Lipschitz neural network with theoretical guarantees for its robustness and expressive power. SortNet generalizes [L-infinity distance net](https://github.com/zbh2047/L_inf-dist-net) and can be efficiently trained to achieve certified L-infinity robustness for free!

[Our paper](https://arxiv.org/abs/2210.01787) has been accepted for NeurIPS 2022 (Oral!).

## Dependencies

- Pytorch 1.10.0 (or a later version)
- Tensorboard (optional)


## Getting Started with the Code

This organization of this repo is the same as [L_inf-dist-net-v2](https://github.com/zbh2047/L_inf-dist-net-v2).

### Installation

After cloning this repo into your computer, first run the following command to install the CUDA extension.

```
python setup.py install --user
```

### Reproducing SOTA results

We provide complete training scripts to reproduce the results in our paper. These scripts are in the `commands` folder. 

For example, to reproduce the result of SortNet+MLP architecture on CIFAR-10 with perturbation eps=2/255, simply run

```
bash command/cifar2_255_mlp.sh
```

We also support training the L-infinity distance net since it is a special case of SortNet. To do so, simply change the command by setting the dropout rate to 1.0.



### Pretrained Models and Training Logs

We also provide pretrained models with SOTA certified robust accuracy. These models can be downloaded [here](https://drive.google.com/drive/folders/1zIcnkcm48jtzndWREmJXLhqWtWtcG9wk?usp=sharing). To use these models, follow the **Saving and Loading** instruction above.

Besides, we provide complete training logs for all models and datasets used in paper. They can be found in the `logs` folder.


## Advanced Training Options

### Multi-GPU Training

We also support multi-GPU training using distributed data parallel. By default the code will use all available GPUs for training. To use a single GPU, add the following parameter `--gpu GPU_ID` where `GPU_ID` is the GPU ID. You can also specify `--world-size`, `--rank` and `--dist-url` for advanced multi-GPU training.

### Saving and Loading

The model is automatically saved when the training procedure finishes. Use `--checkpoint model_file_name.pth` to load a specified model before training. You can use `--start-epoch NUM_EPOCHS` to skip training and only test the model's performance for a pretrained model, where `NUM_EPOCHS` is the number of epochs in total.

### Displaying training curves

By default the code will generate five files named `train.log`, `test.log`,  `train_inf.log`, `test_inf.log`and `log.txt` which contain all training logs. If you want to further display training curves, you can add the parameter `--visualize` to show these curves using Tensorboard. 


## Contact

Please contact [zhangbohang@pku.edu.cn](zhangbohang@pku.edu.cn)  if you have any question on our paper or the codes. Enjoy! 



## Citation

```
@inproceedings{zhang2022rethinking,
      title={Rethinking Lipschitz Neural Networks and Certified Robustness: A Boolean Function Perspective}, 
      author={Bohang Zhang and Du Jiang and Di He and Liwei Wang},
      booktitle={Advances in Neural Information Processing Systems},
      year={2022},
}
```

