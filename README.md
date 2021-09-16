# Gradient Boosting Bayesian Networks with MCMC

This is the repository containing the source code relating to this research:

== insert paper ==

## Requirements

The code was implemented in Python 3.6.10 and utilized the following important packages:

```bash
cudatoolkit=11.1.1
numpy=1.19.5
pandas=1.1.5
python=3.6.10
pytorch=1.9.0
```

## Configuration

You can run the experiments with different parameter configurations by specifying them in the `config.yaml` file.

```yaml
params:
  lr: 0.005             # Learning rate
  num_nets: 3           # Number of weak learners
  data: "sunspot"       # Target dataset
  exps: 30              # Number of total experiments to run
  cuda: true            # Use CUDA or not
```

## Experiments

To run the code, simply navigate into the `src/` folder and run the relevant python script

```bash
cd src

python run_sequential.py
```
