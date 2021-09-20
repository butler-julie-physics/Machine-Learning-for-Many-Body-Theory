import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import math
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
from scipy import interp, arange, exp


from CCD_PairingModel import *
from MBPT_PairingModel import *
from DataSets import *
sys.path.insert(1, '/Users/juliehartley/Machine-Learning-for-Many-Body-Theory/SRE')
from Regression import *
from Analysis import *
from Extrapolate import *
from Support import *
sys.path.insert(1, '/Users/juliehartley/Machine-Learning-for-Many-Body-Theory/MBPT-KRR')
from DataSplit import *

def rmse(predictions, targets):
    return math.sqrt(np.mean((predictions-targets)**2))

data_name, training_dim, X_tot, y_tot = VaryDimensionLong()
training_dim = 11
Xtot = X_tot
ytot = y_tot
print((Xtot))

best_err = 100
best_seq = None
best_dim = None
best_auto = None
best_numRetrain = None
best_params = None
for dim in [11]:
    for seq in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for auto in [True, False]:
            for numRetrain in [1, seq]:
                for alpha in [0, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1]:
                    for gamma in np.arange(-10, 10, 0.5):
                            for c0 in np.arange(-10, 10, 0.5):
                                for degree in [1, 2, 3, 4]:
                                    #dim = training_dim
                                    ytrain = y_tot[:dim].tolist()
                                    # Note: training data needs to be a LIST
                                    #ytrain = ytrain.tolist()
                                    length = len(y_tot)

                                    # Regression Parameters
                                    #seq = 3
                                    alpha = 0

                                    # Format only the y component of the training data using sequential
                                    # data formatting
                                    X1, y1 = format_sequential_data (ytrain, seq=seq)
                                    params = [gamma, c0, degree]

                                    krr = KRR(params, 'p', alpha)
                                    try:                                    
                                        krr.fit(X1, y1)
                                        y_pred = sequential_extrapolate(krr, ytrain, len(ytot), seq=seq,\
                                            isAutoRegressive = auto, numRetrain = numRetrain, isErrorAnalysis = False, y_true = ytot)
                                    except:
                                        print("Singular Matrix")
                                        pass
                                    err_sre_krr = rmse(np.asarray(y_tot), np.asarray(y_pred))

                                    if err_sre_krr < best_err:
                                        best_err = err_sre_krr
                                        best_seq = seq
                                        best_dim = dim
                                        best_auto = auto
                                        best_numRetrain = numRetrain
                                        best_params = params

                                        print("*******************", best_err, best_seq, best_dim, best_auto, best_err**2)            
                                        
print("*******************", best_err, best_seq, best_dim, best_auto, best_numRetrain, best_params)            