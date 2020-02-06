# People's Reinforcement Learning (PRL)

![](https://img.shields.io/badge/python-3.6-blue.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)
![](https://readthedocs.org/projects/prl/badge/?version=latest)

## Description

This is a reinforcement learning framework made with research activity in mind.
You can read mode about PRL in our 
[blogpost](https://medium.com/asap-report/prl-a-novel-approach-to-building-a-reinforcement-learning-framework-in-python-208cb8ae9349?sk=ea595f44fc8bd3f2aa4416c997d16891), 
[documentation](https://prl.readthedocs.io/en/latest/index.html) or
[wiki](https://gitlab.com/opium-sh/prl/wikis/home).

## System requirements

* ```python 3.6```
* ```swig```
* ```python3-dev```

We recommend using ```virtualenv``` for installing project dependencies.

## Installation

* clone the project:

  ```
  git clone git@gitlab.com:opium-sh/prl.git
  ```

* create and activate a virtualenv for the project (you can skip this step if you are not using virtualenv)

  ```
  virtualenv -p python3.6 your/path && source your/path/bin/activate
  ```

* install dependencies:

  ```
  pip install -r requirements.txt
  ```
  
* install library

  ```
  pip install -e .
  ```

* run example:

   ```
   cd examples
   python cart_pole_example_cross_entropy.py
   ```