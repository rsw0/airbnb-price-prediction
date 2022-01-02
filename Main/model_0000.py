from typing import Tuple
import numpy as np
import sys
import pandas as pd
import time
from collections import Counter

class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "Runsheng Wang", "BU_ID": "U32829925", "BU_EMAIL": "runsheng@bu.edu"}
    SELECT_COL = None
    MY_MODEL = None

    def __init__(self):
        self.theta = None

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        if Model.SELECT_COL is None:
            # record preprocess time
            start_time = time.time()
            # get input data into pandas to use pandas methods
            X_df = pd.DataFrame(X)
            # note: manually inspected dataset to identify the index where the Location data started
            # from observation, location data starts at index 213 and ends on 762, both inclusive (use 213 and 763 on pandas zero-indexing)
            # some features in there are not location data. I would compare similarity to determine if
            # two columns should be combined. I would also eliminate columns with very few positive observations
            # thus, it is very unlikely that these approach would change the outcome of non-location features
            location_subset_index = np.arange(213,763)
            remaining_subset_index = np.arange(0,213)
            X_df_location = X_df[location_subset_index]
            X_df_remaining = X_df[remaining_subset_index]
            # find the ratio of class 0 to class 1 in each column, remove those that are rare (not much information in them)
            # cutoff is set to be 0.01% of all the rows. If one category is 95% or more of the entire column, then discard that column
            # as it wouldn't help much in predicting. The choice of 0.01 was chosen because given this specific amount of training data
            # 0.01% almost amounts to a single observation
            cutoff = round(len(X_df_location)*0.0001)
            def my_counter(col_index):
                counts = Counter()
                for count in X_df[col_index]:
                    counts[count] += 1
                if counts[0]<cutoff or counts[1]<cutoff:
                    return False
                else:
                    return True
            # cutoff dict stores the column names (or indices in the original X_df dataframe) as keys and a boolean value as value
            cutoff_dict = {}
            for (columnName, columnData) in X_df_location.iteritems():
                cutoff_dict[columnName] = my_counter(columnName)
            # remove those columns, but save a list of values right here for counting purposes later
            tf_list = list(cutoff_dict.values())
            cutoff_dict = {key:value for (key, value) in cutoff_dict.items() if value == True}
            filtered_indices = np.asarray(list(cutoff_dict.keys()))
            X_df_location = X_df_location[filtered_indices]
            # count the number of features removed by this approach
            num_false = 0
            for q in range(len(tf_list)):
                if tf_list[q] == False:
                    num_false += 1
            # line below prints out the number of features removed using the 0.01% procedure
            # print("a total number of " + str(num_false) + " features were removed")
            # count the features that are 
            temp_counter = 0
            for (columnName1, columnData1) in X_df_location.iteritems():
                for (columnName2, columnData2) in X_df_location.iteritems():
                    if pd.Series.equals(columnData1, columnData2):
                        temp_counter += 1
            # line below prints the temp_counter variable. On this training set, 268 features were selected from the 213 to 763
            # features, if tempcounter equal to the number of features in the updated X_df_location, then it means
            # no two columns are exactly the same (as each feature is the same to itself). 
            # There might still be colinearity on the categorical features, but regularization should sort them out. 
            # Note that by running it on 213-762 feature results in 6871 feature being the same. I have removed them
            # in the previous step. Since I manually checked that all these features are non-overlapping, proceed to train the regressor
            # print(temp_counter)
            # Combine the original portion of the dataset with the modified portion of the locations dataset
            # Join does the join based on index only. If you want more sophisticated ways, use merge
            X_df = pd.DataFrame.join(X_df_remaining, X_df_location)
            # get column names of the preprocessed dataset and change the class variable
            preprocessed_col = np.asarray(list(X_df.columns))
            Model.SELECT_COL = preprocessed_col
            # convert X to numpy array afterwards, remember how you converted as well
            X = X_df.to_numpy()
            print("--- training preprocessing took %s seconds ---" % (time.time() - start_time))
        else:
            start_time = time.time()
            X_df = pd.DataFrame(X)
            X_df = X_df[Model.SELECT_COL]
            X = X_df.to_numpy()
            print("--- testing preprocessing took %s seconds ---" % (time.time() - start_time))
        return X, y

    class ElasticNet:
        # implementated Elastic Net (combined L1 and L2 penalty)
        # Elastic Net has two lambdas that can be controlled separaterly, but for ease of cross validation
        # I used the alpha parameter approach where the sum of the two lambdas equal to 1. This means when
        # l1 is 0.5, l2 is also 0.5. When l1 is 0.2, l2 would be 0.8. This approach limits the possible set
        # of values that can be checked (constrained by the relationship l1 + l2 = 1), but it allows grid
        # search/random search to tune one less hyperparameters. The range/set of values can be adjusted by
        # adjusting the alpha parameter. Alternatively, I could also adjust the equation governing the 
        # relationship between the two. The mix ratio determining the amount of l1 vs l2 regularization to add.  
        # A value of 0 is equivalent to l2 regularization and a value of 1 is equivalent to l1 regularization.
        # Constructor
        def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.005, iterations=500):
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.alpha = alpha
            self.l1_ratio = l1_ratio

        # Fit func for model training
        def fit(self, X, Y):
            # m is the nuber of observations
            self.m = X.shape[0]
            # n is the number of features
            self.n = X.shape[1]
            # W is the weights/parameter values
            self.W = np.zeros(self.n)
            # b is the bias
            self.b = 0
            # loading X and Y into the function as class objects
            self.X = X
            self.Y = np.squeeze(Y)
            # GD starts here
            for i in range(self.iterations):
                self.compute_gradients()
            return self

        # Make predictions using the hypothesis function h(x)
        def predict(self, X) :    
            return np.dot(X,self.W) + self.b
        
        # performs GD/update weights
        def compute_gradients(self):
            # get prediction with current weights
            Y_pred = self.predict(self.X)	
            # initialize the weights to be an array of 0
            dW = np.zeros(self.n)
            for cur_col in range(self.n):
                # check the sign of the gradient for l1 penalty
                # penalty should be negative on the l1 norm if the weight is less than 0
                # for the grtadient computation, first find the difference between the actual and predicted value,
                # first multiply the constant, then find the dot product of the current column of X with the error (self.Y-Y_pred)
                # then add/subtract the l1 penalty depending on the sign of the weight, and then add the l2 penalty
                if self.W[cur_col]>0:
                    dW[cur_col] = -((2/self.m)*(np.dot(((self.X[:,cur_col]).T),(self.Y-Y_pred))) + (self.alpha*self.l1_ratio*np.linalg.norm([self.W[cur_col]],ord=1)) 
                    + (0.5*self.alpha*(1-self.l1_ratio)*np.linalg.norm([self.W[cur_col]],ord=2)))
                else:
                    dW[cur_col] = -((2/self.m)*(np.dot(((self.X[:,cur_col]).T),(self.Y-Y_pred))) - (self.alpha*self.l1_ratio*np.linalg.norm([self.W[cur_col]],ord=1)) 
                    + (0.5*self.alpha*(1-self.l1_ratio)*np.linalg.norm([self.W[cur_col]],ord=2)))
            # updates bias term
            db = - 2*np.sum(self.Y-Y_pred)/self.m
            # update weight and bias with the GD formula
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db
            return self

    def train(self, X_train: np.array, y_train: np.array):
        start_time = time.time()
        # Grid search function that finds best hyperparameters using cross validation
        # Setting a very high MSE at the beginning
        # establish the grid search you want to try, or random search
        # do train_test split, 4 splits, fix them, use them to test all the hyperparameters
        # pick a set of hyperparameters at random or by grid
        # run 4 fold cross validation average the resulting MSE
        # compare this MSE to the current best MSE
        # if lower than current best MSE, record the hyperparameters, not the features
        # continue with the next set of hyperparameters, until exhaust all of the grid.
        def grid_search_cv(X_cv, y_cv):
            # initialize large mse
            best_mse = 10e100
            # define the grid search list
            l1_ratio_list = [0.075, 0.1]
            learning_rate_list = [0.001]
            # storing best hyperparameters, initialize to None
            best_l1 = None
            best_learning_rate = None
            # make the full dataset into four splits
            X_and_y_cv = np.concatenate((X_cv,y_cv),axis=1)
            np.random.shuffle(X_and_y_cv)
            row_split_index = round((X_and_y_cv.shape[0])*0.75)
            X_and_y_cv_train = X_and_y_cv[0:row_split_index,:]
            X_and_y_cv_test = X_and_y_cv[row_split_index:,:]
            X_cv, y_cv = np.hsplit(X_and_y_cv_train, [-1])
            X_validate, y_validate = np.hsplit(X_and_y_cv_test, [-1])
            # loop through all sets of hyperparameter values
            loop_num_count = 0
            for my_l1 in l1_ratio_list:
                print("testing l1 = " + str(my_l1))
                for my_learning_rate in learning_rate_list:
                    # create a model from the given set of parameters, fit it, and create a prediction
                    temp_model = model = Model.ElasticNet(l1_ratio=my_l1, learning_rate=my_learning_rate)
                    temp_fitted_model= temp_model.fit(X_cv, y_cv)
                    temp_pred = temp_fitted_model.predict(X_validate)
                    # compute mean squared error between the predicted array and the actual array
                    temp_mse = np.square(np.subtract(y_validate,temp_pred)).mean()
                    if temp_mse < best_mse:
                        best_l1 = my_l1
                        best_learning_rate = my_learning_rate
                    loop_num_count += 1
                    print(loop_num_count)
            # thought about minibatch, didn't use it because noise is induced from the KFold cross validation
            return best_l1, best_learning_rate
        # run grid_search
        best_l1, best_learning_rate = grid_search_cv(X_train,y_train)
        print(best_l1)
        print(best_learning_rate)
        # Define Model (pass in parameters to modify defaults)
        model = Model.ElasticNet(l1_ratio=best_l1, learning_rate=best_learning_rate, iterations=2000)
        # Training on the full set
        fitted_model = model.fit(X_train, y_train)
        # Setting to class variable
        Model.MY_MODEL = fitted_model
        print("--- training took %s seconds ---" % (time.time() - start_time))

    def predict(self, X_val: np.array) -> np.array:
        start_time = time.time()
        # Prediction on test set
        Y_pred = Model.MY_MODEL.predict(X_val)
        print( "The parameters are ", sorted(list(Model.MY_MODEL.W)))
        print( "The intercept is ", round(Model.MY_MODEL.b, 2 ) )
        print("--- predicting took %s seconds ---" % (time.time() - start_time))
        return Y_pred




    def unused():
        '''
        # function for storing unused code that I wrote
        '''

        '''
        # use np.any to find the columns that are continuous, then use nonzero() to find the indices of these columns
        # then store it in the variable is_continuous_index. Turned tuple into array, then squeezed the dimension
        continuous_index = np.squeeze(np.asarray(np.any((X<1.0)&(X>0.0), axis=0).nonzero()))
        # subset only the continuous values from the original numpy matrix 
        # then use np.squeeze to remove the extra dimension/axe of length 1
        X_cont = np.squeeze(X[:, continuous_index])
        # subset the discrete values by creating an arange array and removing the continuous indices
        discrete_index = np.delete(np.arange(X.shape[1]), continuous_index)
        X_disc = np.squeeze(X[:, discrete_index])
        # Computing Pearson's Correlation Coefficients for Feature Selection (Continuous Features)
        # computing the Pearson product-moment correlation coefficients
        # notice that we added y as an additional input so that it can be included
        # a lot of the pairwise computation is unnecessary, but the pandas result contains a lot of NaN for no reason
        # used numpy instead. rowvar=False treats each column as a variable (the default is row vectors)
        # extract the last element from this matrix, which is the correlation between each variable and price
        # also removed the last element, which is the correlation of price with itself
        # made absolute value because want to filter by positive
        cont_coef_matrix = np.abs(np.corrcoef(X_cont, y, rowvar=False)[-1][:-1])
        # make dictionary to identify which columns are significant, keyed by column index, value is significance
        cont_dict = dict(zip(continuous_index, cont_coef_matrix))
        # filter cont_dict to only include those that have absolute value of the correlation greater than 1
        cont_dict = {key:value for (key, value) in cont_dict.items() if value >= 0.1}
        continuous_index_filtered = cont_dict.keys()
        '''
