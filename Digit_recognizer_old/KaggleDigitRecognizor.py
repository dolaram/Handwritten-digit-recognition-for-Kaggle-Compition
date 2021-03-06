import numpy as np
import csv 
#np.set_printoptions(threshold=10000000)
#np.set_printoptions(precision=3)
#np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1) 
reader=csv.reader(open("trainnew.csv"),delimiter=",") 
print("reading is done")
x=list(reader)
result=np.matrix(x).astype("float")
print("Up to here things  are all right")
y=result[:,0]
X=result[:,1:757]/255
print(np.shape(X))
print("Everything is All right")
y=y.astype("int")

import numpy as np
from function import randInitializeWeights
from function import sigmoidal
from function import nnCostFunction
from function import nngradfunction
from scipy import optimize
# Setting the precision and threshold for printing the full matrix
np.set_printoptions(threshold=10000000)
np.set_printoptions(precision=6)

# Setup the parameters you will use for this exercise
input_layer_size  = 756;  # 20x20 Input Images of Digits
hidden_layer_size = 70;   #25 hidden units
num_labels = 10;          #10 labels, from 1 to 10 

(yrow,col)=np.shape(y)

for i in range(yrow):
    if y[i]==10:
        y[i]=0

(m,col)=np.shape(X)
# ================ Part 1: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)
print('\nInitializing Neural Network Parameters .....')
initial_Theta1 = np.matrix(randInitializeWeights(input_layer_size, hidden_layer_size))
initial_Theta2 = np.matrix(randInitializeWeights(hidden_layer_size, num_labels))
#% Unroll parameters
initial_nn_params = np.vstack( ( np.transpose(initial_Theta1.flatten()),np.transpose(initial_Theta2.flatten()) ) )
row=len(initial_nn_params)
initial_nn_params =np.squeeze(np.asarray(initial_nn_params ))
# =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.

print('\nTraining Neural Network............')
#  You should also try different values of lembda
lembda = 0;
#Create "short hand" for the cost function to be minimized
costfunction=lambda nn_params:nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lembda)
gradfunction=lambda nn_params:nngradfunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lembda)

# Now, costFunction is a function that takes in only one argument (the
#neural network parameters)
agrs=[input_layer_size,hidden_layer_size,num_labels,np.matrix(X), np.matrix(y), lembda]
result=optimize.fmin_cg(costfunction,initial_nn_params,fprime=gradfunction,args=(),gtol=1e-05,norm=np.inf,epsilon=1.5e-05, maxiter=1)
print("Optimization end here! Woh")

Theta1 =result[0:hidden_layer_size * (input_layer_size + 1)].reshape([hidden_layer_size,(input_layer_size + 1)])
Theta2 =result[hidden_layer_size*(input_layer_size + 1):row].reshape([num_labels,hidden_layer_size+1])


reader=csv.reader(open("testnew.csv"),delimiter=",") 
print("reading is done")
x=list(reader)
result=np.matrix(x).astype("float")
print("Up to here things  are all right")
X=result/255
print(np.shape(X))
print("Everything is All right")

(m,col)=np.shape(X)
p=np.zeros((m,1))
h1=sigmoidal(np.column_stack((np.ones((m, 1)), X))*np.transpose(Theta1))
h2=sigmoidal(np.column_stack((np.ones((m, 1)), h1))*np.transpose(Theta2))
p=h2.argmax(1)

#print("Accuracy of the ANN is =", np.mean(p==y)*100,"%")
print(np.shape(initial_nn_params))

a=np.concatenate((np.transpose(np.matrix(range(1,len(p)+1))),p), axis=1).astype("int")
np.savetxt("testoutnew.csv", a, delimiter=",")
print("ANN stops here")








