#This script creates train/val/test split csv files for each data type that we plan to experiment with

from config import cfg
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


