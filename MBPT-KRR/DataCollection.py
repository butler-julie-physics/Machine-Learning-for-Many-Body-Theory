# A basic root mean squared function for error analysis
def rmse(predictions, targets):
    return math.sqrt(np.mean((predictions-targets)**2))

#
# Define the sigma value of the Gaussian function f and then create
# the TATRs
sigmaT = 0.5
dx = 0.5
TATRs = []
X = 0
for amp in amps:
    TATRs.append(TATR(dx, amp, sigmaT))
Xdata = TATRs

# Define the value of sigma to be used by the KRR algorithm and initialize
# the algorithm
sigma = 0.5
params1 = [sigma]
krr = KRR(params1, 'gaussian', alpha = 1e-5)

# This code cell splits the data into training and test sets so
# that the test set will always come after the training set.
# This makes this into a extrapolation problem.

# Define the dividing point dim.  Points with an index less than dim will
# be the training set and points with an index greater than or equal to dim
# will be the test set
dim = 20
X_train = Xdata[:dim] 
X_test = Xdata[dim:] 
y_train = corr2[:dim] 
y_test = corr2[dim:]

# Fit the KRR model
krr.fit(X_train,y_train)

# Use the trained model to predict the test data set and report the RMSE
y_pred = krr.predict(X_test)
print("******Test RMSE******", rmse(np.asarray(y_pred), np.asarray(y_test)))
print("******Test MSE******", rmse(np.asarray(y_pred), np.asarray(y_test))**2)

# Use the trained model to predict the whole data set and graph the results
y_pred = krr.predict(Xdata)
plt.plot(ga, corr2, label='True Data', linewidth=3)
plt.plot(ga, y_pred, ':', label='Predicted Data', linewidth=3)
plt.axvspan(ga[0], ga[dim], alpha=0.25, label='Training Data')
plt.legend()    