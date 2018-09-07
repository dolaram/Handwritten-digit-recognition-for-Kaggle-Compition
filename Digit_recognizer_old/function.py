import numpy as np

def sigmoidal(z):
    result=1/(1+np.exp(-z))
    return result;

def randInitializeWeights(L_in,L_out):
    W=np.zeros((L_out,L_in+1))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init- epsilon_init
    return W;

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lembda):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    #for our 2 layer neural network
    row=len(nn_params)
    (m,col)=np.shape(X)
    Theta1 =nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape([hidden_layer_size,(input_layer_size + 1)])
    Theta2 =nn_params[hidden_layer_size*(input_layer_size + 1):row].reshape([num_labels,hidden_layer_size+1])

    # Setup some useful variables
    (m,col)=np.shape(X)
    # You need to return the following variables correctly 
    Theta1_grad = np.matrix(np.zeros(np.shape(Theta1)))
    Theta2_grad = np.matrix(np.zeros(np.shape(Theta2)))
    J = 0
    delta2=np.matrix(np.zeros((hidden_layer_size,1)))
    delta3=np.matrix(np.zeros((num_labels,1)))
    for i in range(m):
            Y=np.matrix(np.zeros((num_labels,1)))
            a1=np.vstack(([1],np.transpose(X[i,:])))
            a20=np.matrix(sigmoidal(Theta1*a1))
            a2=np.vstack(([1],a20)) 
            a3=np.matrix(sigmoidal(Theta2*a2)) 
            Y[y[i]]=1
            J=J+(-np.transpose(Y)*np.log(a3) - np.transpose(1-Y)*np.log(1-a3) )
            delta3=a3-Y
            delta2=np.multiply(np.transpose(Theta2)*delta3,np.multiply(a2,(1-a2)))
            Theta1_grad=Theta1_grad + delta2[1:len(delta2)] * np.transpose(a1)
            Theta2_grad=Theta2_grad + delta3 * np.transpose(a2)
            # for loop end here
    (Theta1_rows,Theta1_cols)=np.shape(Theta1)
    (Theta2_rows,Theta2_cols)=np.shape(Theta2)
    J=J/m + (np.sum(np.multiply(Theta1[:,1:Theta1_cols],Theta1[:,1:Theta1_cols])) +\
         np.sum(np.multiply(Theta2[:,1:Theta2_cols],Theta2[:,1:Theta2_cols])))*lembda/(2*m)
    Theta1_grad=Theta1_grad/m
    Theta2_grad=Theta2_grad/m
    Theta1_grad[:,1:Theta1_cols]=Theta1_grad[:,1:Theta1_cols]+lembda*Theta1[:,1:Theta1_cols]/m
    Theta2_grad[:,1:Theta2_cols]=Theta2_grad[:,1:Theta2_cols]+lembda*Theta2[:,1:Theta2_cols]/m
    # Unroll gradients
    grad = np.vstack((np.transpose(Theta1_grad.flatten()), np.transpose(Theta2_grad.flatten())))
    return J#, grad
#this function ends here
    

def nngradfunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lembda):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    #for our 2 layer neural network
    row=len(nn_params)
    (m,col)=np.shape(X)
    Theta1 =nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape([hidden_layer_size,(input_layer_size + 1)])
    Theta2 =nn_params[hidden_layer_size*(input_layer_size + 1):row].reshape([num_labels,hidden_layer_size+1])

    # Setup some useful variables
    (m,col)=np.shape(X)
    # You need to return the following variables correctly 
    Theta1_grad = np.matrix(np.zeros(np.shape(Theta1)))
    Theta2_grad = np.matrix(np.zeros(np.shape(Theta2)))
    J = 0
    delta2=np.matrix(np.zeros((hidden_layer_size,1)))
    delta3=np.matrix(np.zeros((num_labels,1)))
    for i in range(m):
            Y=np.matrix(np.zeros((num_labels,1)))
            a1=np.vstack(([1],np.transpose(X[i,:])))
            a20=np.matrix(sigmoidal(Theta1*a1))
            a2=np.vstack(([1],a20)) 
            a3=np.matrix(sigmoidal(Theta2*a2)) 
            Y[y[i]]=1
            J=J+(-np.transpose(Y)*np.log(a3) - np.transpose(1-Y)*np.log(1-a3) )
            delta3=a3-Y
            delta2=np.multiply(np.transpose(Theta2)*delta3,np.multiply(a2,(1-a2)))
            Theta1_grad=Theta1_grad + delta2[1:len(delta2)] * np.transpose(a1)
            Theta2_grad=Theta2_grad + delta3 * np.transpose(a2)
            # for loop end here
    (Theta1_rows,Theta1_cols)=np.shape(Theta1)
    (Theta2_rows,Theta2_cols)=np.shape(Theta2)
    J=J/m + (np.sum(np.multiply(Theta1[:,1:Theta1_cols],Theta1[:,1:Theta1_cols])) +\
         np.sum(np.multiply(Theta2[:,1:Theta2_cols],Theta2[:,1:Theta2_cols])))*lembda/(2*m)
    Theta1_grad=Theta1_grad/m
    Theta2_grad=Theta2_grad/m
    Theta1_grad[:,1:Theta1_cols]=Theta1_grad[:,1:Theta1_cols]+lembda*Theta1[:,1:Theta1_cols]/m
    Theta2_grad[:,1:Theta2_cols]=Theta2_grad[:,1:Theta2_cols]+lembda*Theta2[:,1:Theta2_cols]/m
    # Unroll gradients
    grad = np.vstack((np.transpose(Theta1_grad.flatten()), np.transpose(Theta2_grad.flatten())))
    return  np.squeeze(np.asarray(grad))
#this function ends here
def randInitializeWeights1(L_in,L_out):
    w=np.matrix(np.random.randn(L_out, 1 + L_in))
    w=w-w.mean(1)
    w=w/np.sqrt(L_in+1)
    return w



    