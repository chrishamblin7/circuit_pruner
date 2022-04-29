# CIRCUIT PRUNER




## Setup

First things first for all setup methods is to clone this repo:

`git clone https://github.com/chrishamblin7/circuit_pruner_cvpr2022.git`

### Conda

Enter into the downloaded git repo: `cd circuit_pruner_cvpr2022`

Create a new conda environment from the environment_file: `conda env create -f environment/environment.yml`

Activate the new "circuit_pruner" environment: `conda activate circuit_pruner`

Add the circuit_pruner package itself to the "circuit_pruner" environment: `pip install -e .`

Then, if you want to use jupyter notebooks, add the viscnn environment as a kernel with: `python -m ipykernel install --user --name=circuit_pruner`
