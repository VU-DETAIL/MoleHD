# MoleHD
MoleHD: Automated Drug Discovery using Brain-Inspired Hyperdimensional Computing

**Note: If you find my work useful, please use the approproiate citations as below.**

```
(bibtex)
```

If there are any technical questions after the README, please contact:
* dma2#villanova.edu
* connectthapa84@gmail.com

* Dependable, Efficient, and Intelligent Computing Lab (DETAIL)
  	* Dongning Ma (Ph.D. Students, EECS)
	* Rahul Thapa (B.S, CS)
  	* Xun Jiao (Faculty Advisor, EECS)


## Table of Contents
1. [Datasets](#Datasets)
2. [Requirements](#Requirements)
3. [Training/Retraining](#Training)

**Note: If you want to learn more about our work, check out our full paper at [link](link)**

## Datasets

Note: The dataset used in this experiment are already inside data folder. Below, we linked their sources. 

 - Clintox dataset: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz
 - BBBP and Sider dataset: http://moleculenet.ai/datasets-1 
 - Smile-PE: https://github.com/XinhaoLi74/SmilesPE

## Requirements

The conda setup is the official setup for this repo:

Make sure you have conda(anaconda or miniconda) installed

1. cd into repo directory, verify "env.yml" is in current directory
2. Create conda env based on env.yml:  
`$ conda env create --file env.yml -n env_conda`

note: You can replace `env_conda` with any env name you want. 

3. activate the new conda env:  
`$ conda activate env_conda` 
4. to deactivate your environment:   
`$ conda deactivate`  

## Training
Make sure you clone this repository and open command prompt with this project as parent directory. 

## Steps for training/retraining
1. We provide you with the MoleHD training script, [MoleHD.py](./MoleHD.py)
2. To train the model, run the following script. Note that all of these parameters has default value. Therefore, if you simply run the script without any parameter, it will give you a version of our result. 
```
python MoleHD.py \
    --dataset_file ./data/clintox.csv \
    --target CT_TOX \
    --mols smiles \
    --num_tokens 1500 \
    --dim 10000 \
    --max_pos 256 \
    --gramsize 2 \
    --retraining_epochs 150 \
    --iterations 5 \
    --test_size 20 \
    --threshold 1024 \
    --encoding_scheme smiles_pretrained \
    --split_type random \
    --version v1
```
