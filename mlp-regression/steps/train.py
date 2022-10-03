"""
This module defines the following routines used by the 'train' step of the regression pipeline:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""


from binhex import LINELEN


def estimator_fn():

    from sklearn.linear_model import LinearRegression
    
    return LinearRegression()
  
  
    # from sklearn.linear_model import SGDRegressor

    # return SGDRegressor(random_state=42)




    # from sklearn import tree
    # return tree.DecisionTreeRegressor()
    
    # from sklearn.neural_network import MLPRegressor
    
    # return MLPRegressor(hidden_layer_sizes=(33,), activation='relu', learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000, solver='adam', verbose=True)
  