# conda install pandas
# conda install seaborn
# conda install scikit-learn
# conda install -c conda-forge xgboost
# conda install -c conda-forge imbalanced-learn
# conda install -c conda-forge pytables
# conda install -c conda-forge pandas-profiling
# conda install plotly
# conda install catboost
# conda install -c conda-forge kmodes
# conda install -c conda-forge lightgbm
# conda install -c conda-forge tensorflow

import config as config
from functions import Functions as fn

import os
import time
import warnings
import collections
import itertools
from enum import Enum

from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as imbpipeline

from kmodes.kprototypes import KPrototypes

from lightgbm import LGBMClassifier

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as pylab


import numpy as np


from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

import plotly.express as px


import phik
from phik import resources, report
from phik.report import plot_correlation_matrix

import random

import scipy.stats as stats
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import seaborn as sns

from scipy.stats import chi2_contingency

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import make_scorer

from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA
from sklearn.utils import class_weight, compute_class_weight

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb
XGBClassifier = xgb.XGBClassifier
#from xgboost import XGBClassifier

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore',category=FutureWarning)



import pandas as pd
import numpy as np
import warnings
from pandas_profiling import ProfileReport
from IPython.display import IFrame
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import tensorflow as tf
from keras import Sequential, layers, regularizers, optimizers, metrics 
from matplotlib.colors import ListedColormap
from keras.wrappers.scikit_learn import KerasClassifier




