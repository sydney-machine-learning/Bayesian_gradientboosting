# Gradient Boosting Bayesian Networks with MCMC

This is the repository containing the source code relating to this research:

== insert paper ==

## TODO

- Do diagnostic plots
- Perform convergence analysis

## Requirements

The code was implemented in Python 3.6.10 and utilized the following important packages:

```bash
cudatoolkit=11.1.1
numpy=1.19.5
pandas=1.1.5
python=3.6.10
pytorch=1.9.0
```

Install the packages in a conda environment before running.

## Configuration

You can run the experiments with different parameter configurations by specifying them in the `config.yaml` file.

```yaml
params:
  # Classification datasets
  # - ionosphere
  # - cancer
  #
  # Regression datasets
  # - sunspot
  # - rossler
  data: "sunspot"


  lr: 0.01                  # Learning rate
  num_nets: 5               # Number of weak learners to train
  exps: 30                  # Number of experiments to run 
  cuda: true                # Use GPU

  # Train on multistep version on the problem
  # ( Only applies for timeseries regression problems )
  multistep: false

  # Options for MCMC sampling
  mcmc: true                # Use MCMC, below options don't matter if false
  samples: 5000             # Number of samples to generate
  langevin_gradients: true  # Use langevin gradients
  lg_rate: 0.5              # Rate for langevin gradients
```

## Experiments

To run the code, simply navigate into the `src/` folder and run the relevant python script

```bash
cd src

python run_sequential.py
```
