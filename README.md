# CAGPJax

[![Tests](https://github.com/sethaxen/CAGPJax/actions/workflows/run_tests.yml/badge.svg)](https://github.com/sethaxen/CAGPJax/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/sethaxen/CAGPJax/branch/main/graph/badge.svg)](https://codecov.io/gh/sethaxen/CAGPJax)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://sethaxen.github.io/CAGPJax)

Computation-Aware Gaussian Processes for GPJax

## Installation

```bash
pip install git+https://github.com/sethaxen/CAGPJax.git
```

## Simple Example

```python
import jax
import jax.numpy as jnp
import gpjax as gpx
import cagpjax
import optax as ox

# Enable float64 for higher numerical precision
jax.config.update("jax_enable_x64", True)

n_data = 1_000

# Build_model
prior = gpx.gps.Prior(
    mean_function=gpx.mean_functions.Zero(),
    kernel=gpx.kernels.RBF(lengthscale=1.0, variance=1.0),
)
likelihood = gpx.likelihoods.Gaussian(n_data, obs_stddev=0.1)
posterior = prior * likelihood

key = jax.random.key(42)

# Sample data from prior-predictive distribution
key, subkey = jax.random.split(key)
x_train = jnp.linspace(0, 10, n_data).reshape(-1, 1)
y_train = prior.predict(x_train).sample(subkey, (1,)).squeeze(0)
key, subkey = jax.random.split(key)
y_train = (y_train + jax.random.normal(subkey, y_train.shape)).reshape(-1, 1)
train_data = gpx.Dataset(x_train, y_train)

# Condition a CaGP with an (untrained) sparse linear solver policy
key, subkey = jax.random.split(key)
policy = cagpjax.policies.BlockSparsePolicy(n_actions=10, n=n_data, key=subkey)
cagp = cagpjax.models.ComputationAwareGP(posterior, policy)

# Optimize hyperparameters (including actions)
def negative_elbo(cagp, train_data):
    cagp.condition(train_data)  # update intermediates
    return -gpx.objectives.elbo(cagp, train_data)

cagp_optimized, history = gpx.fit(
    model=cagp,
    objective=negative_elbo, 
    train_data=train_data,
    optim=ox.adamw(learning_rate=0.01),
    num_iters=250,
    key=key,
)

# Get CaGP posterior distribution at the inputs
cagp_optimized.condition(train_data)
cagp_post = cagp_optimized.predict()
```

## Citation

There's not yet a citation for this package.
If using the code, please cite

```bibtex
@inproceedings{wenger2022itergp,
  title         = {Posterior and Computational Uncertainty in {G}aussian Processes},
  author        = {Wenger, Jonathan and Pleiss, Geoff and Pf\"{o}rtner, Marvin and Hennig, Philipp and Cunningham, John P},
  year          = 2022,
  booktitle     = {Advances in Neural Information Processing Systems},
  publisher     = {Curran Associates, Inc.},
  volume        = 35,
  pages         = {10876--10890},
  url           = {https://proceedings.neurips.cc/paper_files/paper/2022/file/4683beb6bab325650db13afd05d1a14a-Paper-Conference.pdf},
  editor        = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
  eprint        = {2205.15449},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}
@inproceedings{wenger2024cagp,
  title         = {Computation-Aware {G}aussian Processes: Model Selection And Linear-Time Inference},
  author        = {Wenger, Jonathan and Wu, Kaiwen and Hennig, Philipp and Gardner, Jacob R. and Pleiss, Geoff and Cunningham, John P.},
  year          = 2024,
  booktitle     = {Advances in Neural Information Processing Systems},
  publisher     = {Curran Associates, Inc.},
  volume        = 37,
  pages         = {31316--31349},
  url           = {https://proceedings.neurips.cc/paper_files/paper/2024/file/379ea6eb0faad176b570c2e26d58ff2b-Paper-Conference.pdf},
  editor        = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  eprint        = {2411.01036},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
}
```
