# CAGPJax

Computation-Aware Gaussian Processes (CaGPs) for JAX

CAGPJax provides efficient Gaussian processes by leveraging structured kernel approximations and sparse matrix operations, built on JAX and GPJax.

For $n$ data-points, the computational cost of exact Gaussian processes scales as $\mathcal{O}(n^3)$ due to matrix inversions, while the memory requirements scale as $\mathcal{O}(n^2)$.
CaGPs project the data to a $k(\ll n)$-dimensional subspace to perform inference, reducing the computational cost to $\mathcal{O}(n^2k)$ and the memory requirements to $\mathcal{O}(nk)$.
Using sparse projections further reduces the computational cost to $\mathcal{O}(n^2)$.

Compared to other apprximate GP inference approaches such as inducing point methods, the prediction uncertainty of CaGPs accounts for the additional uncertainty due to only observing a subspace of the data.

## Installation

```bash
pip install git+https://github.com/sethaxen/CAGPJax.git
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

## API Reference

See the [Reference](reference/) section for detailed API documentation.