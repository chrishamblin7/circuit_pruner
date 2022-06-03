
# CIRCUIT PRUNER

## Setup

First things first for all setup methods is to clone this repo:

`git clone https://github.com/chrishamblin7/circuit_pruner.git`


### Conda

Create a new conda environment from the environment_file: `conda env create -f environment/environment.yml`

Activate the new "circuit_pruner" environment: `conda activate circuit_pruner`

Add the circuit_pruner package itself to the "circuit_pruner" environment (run inside the root folder, with the setup.py file): `pip install -e .`

Then, if you want to use the jupyter notebooks, add the viscnn environment as a kernel with: `python -m ipykernel install --user --name=circuit_pruner`


## Experiments

### Methods Comparison

Read the 'notes.md' file in the 'method_comparison' folder. For a quick look at the data run 'mrthod_comparison/plotting.ipynb' and choose the 'quick start' options.

### Subfeatures

for polysemantic experiments: 'polysemantic_subfeatures.ipynb'
for circle-scale experiment:
'circle_subfeatures.ipynb'

## Circuit Diagrams

'circuit_diagram.ipynb'

Use the quick start option to quickly acces the GUI. Will be accessible in a new browser window at 'http://localhost:8050/'

