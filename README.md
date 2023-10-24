# Chappie.ai

[![Languages](https://img.shields.io/github/languages/count/bellerb/chappie.ai?style=flat-square)](#)
[![Top Languages](https://img.shields.io/github/languages/top/bellerb/chappie.ai?style=flat-square)](#)
[![Total Lines](https://img.shields.io/tokei/lines/github/bellerb/chappie.ai)](#)
[![File Size](https://img.shields.io/github/languages/code-size/bellerb/chappie.ai)](#)

## Description

The following is framework for developing reinforcement learning (rl) agents. This bot has been designed as a general framework for performing a multitude of tasks with minimal teaching.

## Folder structure

The folder is layed out like the following.

```
.
├──README.md
├──LICENSE
├──setup.py
├──chappie
├──examples
├──notebooks
```

### Chappie

In the chappie folder is our actual chappie framework. This is the code that is downloaded when you pip install the newly created package.

### Examples

In the examples folder we have examples of how to use the chappie framework. This folder is only included in our github repo and not in the actual package.

### Notebooks

In the notebooks folder we have a series of jupyter notebooks where ideas are flushed out. In these notebooks we go further in detail about each feature being built. This folder is only included in our github repo and not in the actual package.

## Installation Instructions

```cmd
pip install --upgrade chappie-ai
```

## Import Instructions

```python
import chappie
```

## Write Up

To get a better understanding of why the code is written this way check out my detailed write up:

- https://medium.com/@bellerb/playing-chess-with-a-generalized-ai-b83d64ac71fe
- https://medium.com/@bellerb/playing-chess-with-offline-reinforcement-learning-411edc5efd5f
- https://medium.com/@bellerb/retrieval-transformer-enhanced-reinforcement-learning-24509e97c4c6
