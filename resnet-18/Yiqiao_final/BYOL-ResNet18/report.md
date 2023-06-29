# BYOL based on Resnet18 - Implementation
PyTorch implementation of "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" by J.B. Grill et al.

[Link to paper](https://arxiv.org/abs/2006.07733)

This repository includes a practical implementation of BYOL with:
- Benchmarks on vision datasets (**CIFAR-10** for train & val)
- CIFAR-10 could be downloaed at https://drive.google.com/file/d/1oZiZIZWROOynincRSt7c-MRXF_poVJe1/view?usp=sharing
- Support for PyTorch **<= 1.5.0**
- Optimizer: Adam
- Pre-training epochs: 50
- Batch size: 192
- Learning rate: 3e-4
- Checkpoint epochs: 5
- Experimental environment：
Miniconda: conda3
Python: 3.8(ubuntu18.04)
Cuda: 11.1
GPU: RTX 3090(24GB) * 1
CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
RAM: 80GB


This repository includes a practical implementation of BYOL with:
- Benchmarks on vision datasets (**CIFAR-10** for train & val)
- CIFAR-10 could be downloaed at https://drive.google.com/file/d/1oZiZIZWROOynincRSt7c-MRXF_poVJe1/view?usp=sharing
- Support for PyTorch **<= 1.5.0**
- Optimizer: Adam
- Pre-training epochs: 50
- Batch size: 192
- Learning rate: 3e-4
- Checkpoint epochs: 5
- Experimental environment：
Miniconda: conda3
Python: 3.8(ubuntu18.04)
Cuda: 11.1
GPU: RTX 3090(24GB) * 1
CPU: 15 vCPU Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
RAM: 80GB

## Usage
### Pre-training
To run pre-training using BYOL with the default arguments (1 node, 1 GPU), use:
```
python3 main.py
```

Which is equivalent to:
```
python3 main.py --nodes 1 --gpus 1
```
The pre-trained models are saved every *n* epochs in \*.pt files, the final model being `model-final.pt`

My pre-trained model 'yiqiao-byol-model-final.pt' could be downloaded at https://drive.google.com/file/d/11Sq2ynFkBFeh7H7E_B8G0FWQ-TUqRIQr/view?usp=sharing

### Finetuning & Linear classification protocol
Finetuning a model ('linear evaluation') on top of the pre-trained, frozen ResNet model can be done using:
```
python3 logistic_regression.py --model_path=./model-final.pt
```

With `model_final.pt` being file containing the pre-trained network from the pre-training stage.

## Arguments
```
--image_size, default=224, "Image size"
--learning_rate, default=3e-4, "Initial learning rate."
--batch_size, default=42, "Batch size for training."
--num_epochs, default=100, "Number of epochs to train for."
--checkpoint_epochs, default=10, "Number of epochs between checkpoints/summaries."
--dataset_dir, default="./datasets", "Directory where dataset is stored.",
--num_workers, default=8, "Number of data loading workers (caution with nodes!)"
--nodes, default=1, "Number of nodes"
--gpus, default=1, "number of gpus per node"
--nr, default=0, "ranking within the nodes"
```

## Results
### Training 
#### loss
![](BYOL-master/../figs/accuracy%20of%20train.png)
#### Accuracy
![](BYOL-master/../figs/accuracy%20of%20train.png)
### Linear classification protocol (evaluation) 
#### loss
![](BYOL-master/../figs/loss%20of%20test.png)
#### Accuracy
![](BYOL-master/../figs/accuracy%20of%20test.png)
#### final testing performance
The performance (accuracy) of testing the fine-tuned model after throwing away fc layer is 79.96%.

