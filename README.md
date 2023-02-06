# MoleHD
MoleHD: Drug Discovery using Brain-Inspired Hyperdimensional Computing

If you find this repo useful in your research, please consider citing us:
```
@inproceedings{ma2022molehd,
  title={MoleHD: Efficient Drug Discovery using Brain Inspired Hyperdimensional Computing},
  author={Ma, Dongning and Thapa, Rahul and Jiao, Xun},
  booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={390--393},
  year={2022},
  organization={IEEE}
}
```

If there are any technical questions, please contact:
* dma2@villanova.edu
* connectthapa84@gmail.com

## Core Team

* [Dependable, Efficient, and Intelligent Computing Lab (DETAIL)](https://vu-detail.github.io/) at Villanova University
  	* Dongning Ma (Ph.D. Students, ECE)
	* Rahul Thapa (B.S, CS)
  	* Xun Jiao (Assistant Professor, ECE)


## Table of Contents
1. [Datasets](#Datasets)
2. [Requirements](#Requirements)
3. [Training/Retraining](#Training)

## Datasets

 Clintox, BBBP and SIDER datasets can be downloaded at [MoculeNet dataset hub](https://moleculenet.org/datasets-1).


## Requirements

The conda setup is the official setup for this repo:

1. cd into repo directory, verify "env.yml" is in current directory
2. Create conda env based on env.yml:  
`$ conda env create --file env.yml -n env_conda`

note: You can replace `env_conda` with any environment name you want. 

3. activate the new conda env:  
`$ conda activate env_conda` 
4. to deactivate your environment:   
`$ conda deactivate`  

## Training
Make sure you clone this repository and open command prompt with this project as parent directory. 

## Steps for training/retraining
To train the model, run the following script. Note that all of these parameters has default value. Therefore, if you simply run the script without any parameter, it will give you a version of our result. 
```
python MoleHD.py \
    --dataset_file ./data/clintox.csv \
    --target CT_TOX \
    --mols smiles \
    --num_tokens 500 \
    --dim 10000 \
    --max_pos 256 \
    --gramsize 3 \
    --retraining_epochs 20 \
    --iterations 10 \
    --test_size 20 \
    --threshold 256 \
    --encoding_scheme characterwise \
    --split_type random \
    --version v1
```

## Using Smiles-PE Pre-Trained Data

Should you wish to use Smiles-PE pretrained data for tokenization/encoding, you need to download the [pre-trained file] (https://github.com/XinhaoLi74/SmilesPE/blob/master/SPE_ChEMBL.txt) and put it in "./data/"
