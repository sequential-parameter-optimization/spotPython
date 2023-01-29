# spotPython

Sequential Parameter Optimization in Python

# Development


## Styleguide

Follow the Google Python Style Guide from [https://google.github.io/styleguide/pyguide.html]([https://google.github.io/styleguide/pyguide.html).

## Pre commit checks

Before you commit your code, please check that it is "clean". 
To do so, first run [`black`](https://github.com/psf/black) from the projects root directory:

```
$ black .
```

Next, check if [`flake8`](https://flake8.pycqa.org/en/latest/) shows any errors:

```
$ flake8
```

Fix any shown errors before you commit.

# Installation

`pip install spotPython`


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

https://scipy-lectures.org/advanced/optimizing/index.html

## Editor/IDE

* Optional: Install [visualstudio](https://code.visualstudio.com)
* Optional: Install [quarto](https://quarto.org)


## Package Installation

### Configuration Files

* This information is based on [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
* Information is stored in `pyproject.toml` (`setup.py` is not used anymore.)
* A `src` folder is used for the package sources.
* The following files are used for the package building:
   * `pyproject.toml`: see [pyproject.toml](./pyproject.toml). 
   * Important: Follow the instructions from [https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) for including data files (like *.csv, *.tar, etc.). These files can be specified in the following `MANIFEST` file:
     * `MANIFEST`: see [MANIFEST](MANIFEST.in). It describes the data files to be included, e.g.:
       * `include src/spotPython/data/*.rst`
    * `LICENSE`: see [LICENSE](./LICENSE)

### Installation

* Perform the following steps to install the package:
  * Make sure you have the latest version of PyPA’s build installed:
    * `python3 -m pip install --upgrade build`
  * Start the package building process via:  `python3 -m build` 
  * This command should output a lot of text and once completed should generate two files in the `dist` directory.
  * You can use the local `spotPython*.tar.gz` file from the `dist` folder for your package installation with `pip`, e.g.;
  * `python3 -m pip install ./dist/spotPython-0.0.1.tar.gz`


