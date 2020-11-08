import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import scipy
import matplotlib.pyplot as plt
#from  sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
import seaborn as sns
import matplotlib
from sklearn.preprocessing import Binarizer
import sklearn
import math
from sklearn.preprocessing import OneHotEncoder

#Defaults
plt.rcParams['figure.figsize']=(20.0,10.0)
plt.rcParams.update({'font.size':12})
plt.style.use('ggplot')


df1=pd.read_csv(r"data\PVR.csv")
df2=pd.read_csv(r"data\IMAX.csv")


