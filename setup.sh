#!/bin/bash

# Create virtual environment
mamba create -n energysysalternatives python=3.10

# Activate virtual environment
mamba activate energysysalternatives

# Install fine library with from GitHub with all the dependencies
pip install git+https://github.com/lovinduw/mga_in_fine.git#egg=fine