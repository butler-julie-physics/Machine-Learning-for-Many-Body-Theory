#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


infile = open("MassEval2016.dat",'r')
# Read the experimental data with Pandas
Masses = pd.read_fwf(infile, usecols=(2,3,4,6,11),
              names=('N', 'Z', 'A', 'Element', 'Ebinding'),
              widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
              header=39,
              index_col=False)

# Extrapolated values are indicated by '#' in place of the decimal place, so
# the Ebinding column won't be numeric. Coerce to float and drop these entries.
Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
Masses = Masses.dropna()
# Convert from keV to MeV.
Masses['Ebinding'] /= 1000

# Group the DataFrame by nucleon number, A.
Masses = Masses.groupby('A')
# Find the rows of the grouped DataFrame with the maximum binding energy.
Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])

A = Masses['A']
Z = Masses['Z']
N = Masses['N']
Element = Masses['Element']
Energies = Masses['Ebinding']
print(Masses)


# In[15]:


Energies = np.asarray(Energies).tolist()
A = np.asarray(A).tolist()


# In[17]:


plt.plot(A, Energies)


# In[27]:


import sys
import math
sys.path.insert(1, '/Users/juliehartley/Machine-Learning-for-Many-Body-Theory/SRE')
from Regression import *
from Analysis import *
from Extrapolate import *
from Support import *
sys.path.insert(1, '/Users/juliehartley/Machine-Learning-for-Many-Body-Theory/MBPT-KRR')
from DataSplit import *
from KRRRetrain import *


# In[28]:


def rmse(predictions, targets):
    return math.sqrt(np.mean((predictions-targets)**2))


# In[29]:


x_train, x_test, y_train, y_test, g_train, g_test = split_even(A, Energies, A, split=0.8)


# In[ ]:


# Create the training data
Xtrain = x_train
ytrain = y_train
# Note: training data needs to be a LIST
#ytrain = ytrain.tolist()

best_err = 100
best_params = []
for alpha in [0, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1]:
    for seq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for gamma in np.arange(-5, 5.5, 0.5):
            for c0 in np.arange(-5, 5.5, 0.5):
                for p in [1, 2, 3, 4, 5]:
                    # Format only the y component of the training data using sequential
                    # data formatting
                    X, y = format_sequential_data (ytrain, seq=seq)
                    try:
                        # Initialize and instance of the ridge regression classs and train
                        # it using the formatted data
                        R = KRR([gamma, c0, p], 'p', alpha)
                        R.fit(X, y)

                        # Using the trained ridge regression algorithm, extrapolate the
                        # training data set until the total set is the same length as the
                        # total data set
                        y_pred_sre = sequential_extrapolate(R, ytrain, len(A), seq=seq,                            isAutoRegressive = False, numRetrain = numRetrain, isErrorAnalysis = True, y_true = ytot)
                    except:
                        y_pred_sre = np.zeros(len(A))
                        pass
                    err_sre = rmse(np.asarray(Energies), np.asarray(y_pred_sre))
                    if err_sre < best_err:
                        best_err = err_sre
                        best_params = [alpha, seq, gamma, c0, p]

print(best_err, best_params)


# In[ ]:


print(best_err, best_params)


# In[ ]:




