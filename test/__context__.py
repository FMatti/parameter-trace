"""
Context file
============

This file ensures that the working directory is set to the root of the project,
and that all imports are relative to the root of the project.

Every file in this subdirectory should contain the line

>>> import __context__

in order to ensure that the source modules and ML models are imported correctly.
"""

import os
import sys

# How many directories to go up to get to the root directory of the project
root_dist = 1

# Path to root directory of the project
root = __file__
for i in range(root_dist + 1):
    root = os.path.dirname(root)

# Change working directory to root directory of the project
os.chdir(root)

# Append working directory to system path in order to import source modules
sys.path.append(root)