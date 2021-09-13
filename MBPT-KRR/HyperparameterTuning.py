
best_err = 100
best_params = []
for sigmaT in [0.5, 0.01, 0.05, 0.001, 0.005, 0.001]:
    for sigma in [0.5, 0.01, 0.05, 0.001, 0.005, 0.001]:
        for dx in [0.1, 0.25, 0.5, 0.01, 0.01]:
            for alpha in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                #print(sigmaT, sigma, dx, alpha)
                 # Define the sigma value of the Gaussian function f and then create
                # the TATRs
                #sigmaT = 0.001
                #dx = 0.01
                TATRs = []
                X = 0
                for amp in amps:
                    TATRs.append(TATR(dx, amp, sigmaT))
                Xdata = TATRs  
                
                
                # Define the value of sigma to be used by the KRR algorithm and initialize
                # the algorithm
                #sigma = 0.001
                params1 = [sigma]
                krr = KRR(params1, 'gaussian', alpha = alpha)                

                dim = 20
                X_train = Xdata[:dim] 
                X_test = Xdata[dim:] 
                y_train = corr2[:dim] 
                y_test = corr2[dim:]    
                
                # Fit the KRR model
                krr.fit(X_train,y_train)  
                
                # Use the trained model to predict the test data set and report the RMSE
                y_pred = krr.predict(X_test)
                #print("******Test RMSE******", rmse(np.asarray(y_pred), np.asarray(y_test)))
                #print("******Test MSE******", rmse(np.asarray(y_pred), np.asarray(y_test))**2)
                err = rmse(np.asarray(y_pred), np.asarray(y_test))
                if err < best_err:
                    print("New best error", err)
                    best_err = err
                    best_params = [sigmaT, sigma, dx, alpha]
                    print("Best params", best_params)
                    print()
                    # Use the trained model to predict the whole data set and graph the results
                    y_pred = krr.predict(Xdata)
                    plt.plot(ga, corr2, label='True Data', linewidth=3)
                    plt.plot(ga, y_pred, ':', label='Predicted Data', linewidth=3)
                    plt.axvspan(ga[0], ga[dim], alpha=0.25, label='Training Data')
                    plt.legend()                    
print(best_err, best_params)