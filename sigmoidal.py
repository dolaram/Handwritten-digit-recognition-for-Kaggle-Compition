import numpy as np
def sigmoidal(z):
    f=1/(1+np.exp(-z))
    return f

    