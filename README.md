<!--<br/><br/>
<p align="center">
  <img width="450" src="https://github.com/annahedstroem/sanity-checks-revisited/blob/394f166226e4ac415c6534e0e0441d8b3c9258f2/emprt_smprt_logo.png">
<!--<h3 align="center"><b>CoSy: Evaluating Textual Explanations of Neurons with Concept Synthesis</b></h3>
<p align="center">
  PyTorch-->

  </p>-->

This repository contains the code and experiments for the paper **[INSERT HERE](https://openreview.net/forum?id=vVpefYmnsG)** by Kopf et al., 2024. 

<!--[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](anonymous)-->
<!--![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)-->
<!--[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)-->
<!--[![PyPI version](https://badge.fury.io/py/metaquantus.svg)](https://badge.fury.io/py/metaquantus)-->
<!--[![Python package](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)-->
<!--[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](anonymous)-->

## Citation

If you find this work interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@inproceedings{......,
  title={......},
  author={......},
  booktitle={......},
  year={2024},
  url={......}
}
```
This work is in review.

## Repository overview

The repository is organised for ease of use:
- The `src/` folder contains all necessary functions.
- The `nbs/` folder includes notebooks for generating the plots in the paper and for benchmarking experiments.

The CoSY method implementation can be found in the [script](https://github.com/lkopf/cosy/blob/main/src/evaluation.py). 
To reproduce the experiments, and in particular, to generate explanations, it is necessary to take additional steps to (list). See how to run below.

## How to run

#### 0. Collect Explanations with Explanation Methods

TODO: Write instructions on how to set up the following repos:

- [MILAN](https://github.com/evandez/neuron-descriptions)
- [FALCON](https://github.com/NehaKalibhat/falcon-explain)
- [CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)
- [INVERT](https://github.com/lapalap/invert)

#### 1. Collect Activations

Collect activations for your model

```bash
python src/activation_collector.py
```

#### 2. Generate Explanation Images

```bash
torchrun src/image_generator.py --nproc_per_node=3
```

#### 3. Evaluate Explanations

```bash
python src/evaluation.py
```


## Paper highlights 📚

INSERT DESC
</p>
<p align="center">
  <img width="800" src="INSERT_IMAGE"> 
</p>

INSERT DESC OF IMAGE

## Installation

Install the necessary packages using the provided [requirements.txt](https://github.com/lkopf/cosy/blob/main/requirements.txt):

```bash
pip install -r requirements.txt
```

## Package requirements 

Required packages are:

```setup
python>=3.10.1
torch>=2.0.0
quantus>=0.5.0
metaquantus>=0.0.5
captum>=0.6.0
```

### Thank you

We hope our repository is beneficial to your work and research. If you have any feedback, questions, or ideas, please feel free to raise an issue in this repository. Alternatively, you can reach out to us directly via email for more in-depth discussions or suggestions. 

📧 Contact us:
- Laura Kopf: [laura.kopf@gmx.de](mailto:laura.kopf@gmx.de)

Thank you for your interest and support!


