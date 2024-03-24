**********************************************
# Vishal Jain: S2 Coursework - Lighthouse Problem
**********************************************

## Description
This project codes up a solution to the lighthouse problem described in the `lighthouse.pdf`. The solution is implemented in Python.

## Set up
Run the following steps to set up the project:
These steps assume conda is installed on your system. If not, please install it first.
```
git clone git@github.com:VishalJ99/s2_cw.git

cd s2_cw

conda env create -f environment.yml

conda activate stats_cw
```

## Usage
To run the code with the default settings, ensure the previous set up step is completed and use the following command:
```
python src/main.py
``` 
This code will take approximately 2-3 minutes to run.
Optionally, you can specify the following arguments:

```
--n_chains: Number of chains to run. Default is 5.
--n_samples: Number of samples/steps to run each chain. Default is 1e5.
```

For example, to decrease runtime use the following command to run with just 3 chains and 1e4 samples per chain.
```
python src/main.py --n_chains 3 --n_samples 10000
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Author
Vishal Jain
2024-03-12
