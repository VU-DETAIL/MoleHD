# MoleHD
MoleHD: Automated Drug Discovery using Brain-Inspired Hyperdimensional Computing

### Datasets

Note: The dataset used in this experiment are already inside data folder. Below, we linked their sources. 

 - Clintox dataset: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz
 - BBBP and Sider dataset: http://moleculenet.ai/datasets-1 
 - Smile-PE: https://github.com/XinhaoLi74/SmilesPE

### Conda Environment Setup

The conda setup is the official setup for this repo:

Make sure you have conda(anaconda or miniconda) installed

1. cd into repo directory, verify "env.yml" is in current directory
2. Create conda env based on env.yml:  
`$ conda env create --file env.yml -n env_conda`

note: You can replace `env_conda` with any env name you want. 

3. activate the new conda env:  
`$ conda activate env_conda`

4. Run this command from inside the conda environment (Please check the parameter inside MoleHD.py file and change as you want to experiment):
`python MoleHD`  

4. to deactivate your environment:  
`$ conda deactivate`  
