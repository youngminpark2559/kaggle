import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

def load_csv_in_pd(path):
    data=pd.read_csv(path)
    return data
