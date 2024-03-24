**********************************************
# Vishal Jain: S2 Coursework - Lighthouse Problem
**********************************************

## Description
This project codes up a solution to the lighthouse problem described in the `lighthouse.pdf`. The solution is implemented in Python.

## Set up
Run the following steps to set up the project:
These steps assume conda is installed on your system. If not, please install it first.
```
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/S2_Assessment/vj279.git

cd vj279

conda env create -f environment.yml

conda activate stats_cw
```

## Usage
To run the code with the default settings, ensure the previous set up step is completed and use the following command:
```
python src/main.py
``` 

Optionally, you can specify the following arguments:

```
--n_chains: Number of chains to run. Default is 5.
--n_samples: Number of samples/steps to run each chain. Default is 1e5.
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Author
Vishal Jain
2024-03-12
