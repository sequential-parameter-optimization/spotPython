<p align="left">
  <img height="200" src="img/spotLogo.png" alt="spot_logo">
</p>

# spotpython

Sequential Parameter Optimization in Python

* spotpython is a Python version of the well-known hyperparameter tuner SPOT, which has been developed in the R programming environment for statistical analysis for over a decade [bart21i].
* spotpython is a sequential model-based optimization (SMBO) method [BLP05].


# Installation

* Installation can be done with `pip`:

`pip install spotpython`

# spotpython Documentation

* Hyperparameter-tuning Cookbook: A guide for scikit-learn, PyTorch, river, and spotpython. Available at [https://sequential-parameter-optimization.github.io/spotpython/](https://sequential-parameter-optimization.github.io/spotpython/).

* [Bartz-Beielstein (2023). PyTorch Hyperparameter Tuning --- A Tutorial for spotpython (Working Paper)](https://arxiv.org/abs/2305.11930).

    > **Abstract**: The goal of hyperparameter tuning (or hyperparameter optimization) is to optimize the hyperparameters to improve the performance of the machine or deep learning model. spotpython ("Sequential Parameter Optimization Toolbox in Python") is the Python version of the well-known hyperparameter tuner SPOT, which has been developed in the R programming environment for statistical analysis for over a decade. PyTorch is an optimized tensor library for deep learning using GPUs and CPUs. This document shows how to integrate the spotpython hyperparameter tuner into the PyTorch training workflow.  As an example, the results of the CIFAR10 image classifier are used. In addition to an introduction to spotpython, this tutorial also includes a brief comparison with Ray Tune, a Python library for running experiments and tuning hyperparameters. This comparison is based on the PyTorch hyperparameter tuning tutorial. The advantages and disadvantages of both approaches are discussed. We show that spotpython achieves similar or even better results while being more flexible and transparent than Ray Tune.


# spotpython Features

* Some of the advantages of `spotpython` are:

  - Numerical and categorical hyperparameters.
  - Powerful surrogate models.
  - Flexible approach and easy to use.
  - Simple JSON files for the specification of the hyperparameters.
  - Extension of default and user specified network classes.
  - Noise handling techniques.
  - Tensorboard interaction.

# Citation

```bibtex
@ARTICLE{bart23earxiv,
       author = {{Bartz-Beielstein}, Thomas},
        title = "{PyTorch Hyperparameter Tuning -- A Tutorial for spotpython}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Artificial Intelligence, Mathematics - Numerical Analysis, 68T07, A.1, B.8.0, G.1.6, G.4, I.2.8},
         year = 2023,
        month = may,
          eid = {arXiv:2305.11930},
        pages = {arXiv:2305.11930},
          doi = {10.48550/arXiv.2305.11930},
archivePrefix = {arXiv},
       eprint = {2305.11930},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230511930B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


```bibtex
@book{bart21i,
	editor = {Bartz,Eva and Bartz-Beielstein, Thomas and Zaefferer, Martin and Mersmann, Olaf},
	isbn = {ISBN 978-981-19-5169-5},
	keywords = {bartzPublic},
	note = {in print},
	publisher = {Springer},
	title = {{Hyperparameter Tuning for Machine and Deep Learning with R - A Practical Guide}},
	year = {2022}
  url = {https://link.springer.com/book/10.1007/978-981-19-5170-1}
}
```

```bibtex
@inproceedings{BLP05,
	author = {Bartz-Beielstein, Thomas and Lasarczyk, Christian and Preuss, Mike},
	title = {{Sequential Parameter Optimization}},
	booktitle = {{Proceedings 2005 Congress on Evolutionary Computation (CEC'05), Edinburgh, Scotland}},
	date-added = {2016-10-30 11:44:52 +0000},
	date-modified = {2021-07-22 12:12:43 +0200},
	doi = {10.1109/CEC.2005.1554761},
	editor = {McKay, B and others},
	isbn = {0-7803-9363-5},
	issn = {1089-778X},
	pages = {773--780},
	publisher = {{IEEE Press}},
  address = {Piscataway NJ},
	year = {2005},
	url= {http://dx.doi.org/10.1109/CEC.2005.1554761}
  }

```

# Appendix

* This appendix contains some information on how to setup the development environment for spotpython.
Information provided here is not required for the installation of spotpython.

## Styleguide

Follow the Google Python Style Guide from [https://google.github.io/styleguide/pyguide.html]([https://google.github.io/styleguide/pyguide.html).


## Python

* Mac Users: Install [brew](https://brew.sh/index_de)
  * `brew install python` and `brew install graphviz` etc.

* Generate and activate a virtual environment, see [venv](https://docs.python.org/3/library/venv.html), e.g.,
  * `cd ~; python3 -m venv .venv`
  * `source ~/.venv/bin/activate`

### Python mkdocs

* `python -m pip install mkdocs mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index mkdocs-material`
* `mkdocs build`
* `mkdocs serve`
* `http://127.0.0.1:8000/`


### Optimizing/Profiling Code

* [scipy lecture notes: optimizing code](https://scipy-lectures.org/advanced/optimizing/index.html)

## Editor/IDE

* Optional: Install [visualstudio](https://code.visualstudio.com)
* Optional: Install [quarto](https://quarto.org)


## Package Building

### Local Setup

* This information is based on [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
* Information is stored in `pyproject.toml` (`setup.py` is not used anymore.)
* A `src` folder is used for the package sources.
* The following files are used for the package building:
   * `pyproject.toml`: see [pyproject.toml](./pyproject.toml). 
   * Important: Follow the instructions from [https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) for including data files (like *.csv, *.tar, etc.). These files can be specified in the following `MANIFEST` file:
     * `MANIFEST`: see [MANIFEST](MANIFEST.in). It describes the data files to be included, e.g.:
       * `include src/spotpython/data/*.rst`
    * `LICENSE`: see [LICENSE](./LICENSE)

### Local Installation

* Perform the following steps to install the package:
  * Make sure you have the latest version of PyPA’s build installed:
    * `python3 -m pip install --upgrade build`
  * Start the package building process via:  `python3 -m build` 
  * This command should output a lot of text and once completed should generate two files in the `dist` directory.
  * You can use the local `spotpython*.tar.gz` file from the `dist` folder for your package installation with `pip`, e.g.;
  * `python3 -m pip install ./dist/spotpython-0.0.1.tar.gz`


