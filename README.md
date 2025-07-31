# edbo

Experimental Design via Bayesian Optimization: *edbo* is a practical implementation of Bayesian optimization for chemical synthesis.

This is a reboot of the original project with an added [suzuki coupling](https://en.wikipedia.org/wiki/Suzuki_reaction) example of Bayesian Optimization. It also features a conversion from Jupyter notebooks to Marimo notebooks and a few fixes to outdated code in the original library (fixed library will not be published to any software repository).

This project reboot was done for the Institute for Computing in Research

**Original Repository Paper:** Shields, Benjamin J.; Stevens, Jason; Li, Jun; Parasram, Marvin; Damani, Farhan, Janey, Jacob; Adams, Ryan P.; Doyle, Abigail G. "Bayesian Reaction Optimization as A Tool for Chemical Synthesis" Manuscript Accepted.

**Documentation:**

For parity we've kept the doc files incase they are of use to anyone, but we are not hosting the docs on Github Pages. Please see the original repository site for docs.

https://b-shields.github.io/edbo/index.html

## Installation

We'll be using [uv](https://docs.astral.sh/uv/) to easily manage packages and the virtual environment, but any way you have to install the dependencies in `pyproject.toml` should work fine.


(0) Clone the repository

```
git clone https://github.com/InventBoss/edbo.git
cd edbo
```

(1) Setup the virtual environment

```
uv venv
source .venv/bin/activate
```

(2) Install required dependencies

```
uv sync
```

You should now be able to use the base library with any additional code you wrtire to use edbo for. However, if you want to run the existing notebooks in the repository/create new ones, please continue reading the instructions.

### Running Notebooks

(0) Add Marimo package

```
uv pip install marimo
```

(2) Install extra packages to improve Marimo (OPTIONAL)

```
uv pip install ruff pyarrow
```

Now, just `cd` into any folder with a Marimo notebook (look like ordinary Python files) and run it with:
```
marimo edit {file_name}
```

## Building Paper

To build the paper for the Institute for Computing in Research, simply go to the folder with the paper with `cd cir_paper`, and build it with:

```
pdflatex suzuki_coupling.tex
```
