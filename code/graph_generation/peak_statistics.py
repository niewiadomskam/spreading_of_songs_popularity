import random
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go
import seaborn as sns

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from plotly.subplots import make_subplots
from .initial_graphs import preprocessing
from ..data_scraping.get_artist_metadata import get_release_date

filepath = './data/Final data/all_data.csv'
df = pd.read_csv(filepath,encoding='latin-1')
df = preprocessing(df)

