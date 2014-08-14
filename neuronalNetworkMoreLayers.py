
import numpy as np
import random
import scipy.optimize
import scipy.io as sio


## neuronal net cons
INPUT_LAYER_SIZE = 784
#INPUT_LAYER_SIZE = 400
HIDDEN_LAYER1_SIZE = 50
HIDDEN_LAYER2_SIZE = 100
NUM_LABELS = 10 ## the number 0 is ten

## returns the gradient of the sigmoid function
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z));

## aplied sigmoid function to a matrix
def sigmoid(z):
    return np.divide(1.0, np.add(1.0, np.exp(np.negative(z))))
    

## return the predicted label of x given the trained weights
## theta1, theta2
def predict(theta1, theta2, theta3, x):
    m = x.shape[0]
    num_labels = theta2.shape[1]
    X = np.append(np.ones((m,1)), x , 1)

    Z2 = np.dot(X, np.transpose(theta1))
    A2Aux = sigmoid(Z2)
    A2 = np.append(np.ones((m,1)), A2Aux , 1)
    Z3 = np.dot(A2,np.transpose(theta2))
    A3 = sigmoid(Z3)
    A3 = np.append(np.ones((m,1)), A3 , 1)
    Z4 = np.dot(A3,np.transpose(theta3))
    A4 = sigmoid(Z4)
    return np.argmax(A4, axis=1)+1.0

## randomly init the weights of a layer
def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12    
    return np.random.rand(L_out, 1 + L_in)*2*epsilon_init  - epsilon_init

def nn_costfunction2(nn_params,*args):

    input_layer_size, hidden_layer1_size,  hidden_layer2_size, num_labels, X, y, lam = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    
    Theta1 = np.reshape(nn_params[0:HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1)], (HIDDEN_LAYER1_SIZE, (INPUT_LAYER_SIZE + 1)))
    Theta2 = np.reshape(nn_params[((HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1))):((HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1)))+((HIDDEN_LAYER2_SIZE * (HIDDEN_LAYER1_SIZE + 1)))], (HIDDEN_LAYER2_SIZE, (HIDDEN_LAYER1_SIZE + 1)))
    Theta3 = np.reshape(nn_params[((HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1)))+((HIDDEN_LAYER2_SIZE * (HIDDEN_LAYER1_SIZE + 1))):], (NUM_LABELS, (HIDDEN_LAYER2_SIZE + 1)))

    m = X.shape[0] #Length of vector
    X = np.hstack((np.ones([m,1]),X)) #Add in the bias unit

    layer1 = sigmoid(Theta1.dot(np.transpose(X))) #Calculate first layer
    layer1 = np.vstack((np.ones([1,layer1.shape[1]]),layer1)) #Add in bias unit
    layer2 = sigmoid(Theta2.dot(layer1))
    layer2 = np.vstack((np.ones([1,layer2.shape[1]]),layer2)) #Add in bias unit
    layer3 = sigmoid(Theta3.dot(layer2))

    y_matrix = np.zeros([y.shape[0],layer3.shape[0]]) #Create a matrix where vector position of one corresponds to label
    for i in range(m):
        y_matrix[i,y[i]-1] = 1

    #Cost function
    J = (1.0/m)*np.sum(np.sum(-y_matrix.T.conj()*np.log(layer3),axis=0)-np.sum((1.0-y_matrix.T.conj())*np.log(1.0-layer3),axis=0))
    #Add in regularization
    J = J+(lam/(2.0*m))*np.sum(np.sum(Theta1[:,1:].conj()*Theta1[:,1:])+np.sum(Theta2[:,1:].conj()*Theta2[:,1:])+np.sum(Theta3[:,1:].conj()*Theta3[:,1:]))
  
    #Backpropagation with vectorization and regularization
    delta_4 = layer3 - y_matrix.T
    r3 = Theta3[:,1:].T.dot(delta_4)
    z_3 = Theta2.dot(layer1)
    delta_3 = r3*sigmoidGradient(z_3)
    r2 = Theta2[:,1:].T.dot(delta_3)
    z_2 = Theta1.dot(X.T)
    delta_2 = r2*sigmoidGradient(z_2)
    t1 = (lam/m)*Theta1[:,1:]
    t1 = np.hstack((np.zeros([t1.shape[0],1]),t1))
    t2 = (lam/m)*Theta2[:,1:]
    t2 = np.hstack((np.zeros([t2.shape[0],1]),t2))
    t3 = (lam/m)*Theta3[:,1:]
    t3 = np.hstack((np.zeros([t3.shape[0],1]),t3))
 
    Theta1_grad = (1.0/m)*(delta_2.dot(X))+t1
    Theta2_grad = (1.0/m)*(delta_3.dot(layer1.T))+t2
    Theta3_grad = (1.0/m)*(delta_4.dot(layer2.T))+t3

    nn_params = np.hstack([Theta1_grad.flatten(),Theta2_grad.flatten(),Theta3_grad.flatten()]) #Unroll parameters

   
    return J,nn_params

def main():
    ## load training data 
    DATA = np.genfromtxt('train.csv', delimiter=',', skip_header =1,dtype=np.uint8)
    X,y = DATA[:,1:],DATA[:,0]

    ## load test data for the model

    """, skip_footer=40000"""
    TestData = np.genfromtxt('train.csv', delimiter=',', skip_header=1, skip_footer=20000,dtype=np.uint8)
    XTest = TestData[:,1:]
    ytest = TestData[:,0]
    ## change the label 0 with 10
    y = np.where(y!=0, y, 10)
    ytest = np.where(ytest!=0, ytest, 10)

    
    ## load test data source
##    mat_contentsXy = sio.loadmat('ex4data1.mat')
##    mat_contentsThetas = sio.loadmat('ex4weights.mat')    
##    X = mat_contentsXy['X']
##    y = mat_contentsXy['y']
##    #Theta1 = mat_contentsThetas['Theta1']
##    #Theta2 = mat_contentsThetas['Theta2']

    print X.shape
    print y.shape    

    # test the cost function
##    lambda_training = 1.0
##    init_nn_params = np.hstack([Theta1.flatten(),Theta2.flatten()])
##    args = (INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, NUM_LABELS, X, y, lambda_training)
##
##    res = nn_costfunction(init_nn_params,*args)
##    print res[0]
    

    ##Training call
    lambda_training = 1.0
    initial_Theta1 = randInitializeWeights(INPUT_LAYER_SIZE, HIDDEN_LAYER1_SIZE);
    initial_Theta2 = randInitializeWeights(HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE);
    initial_Theta3 = randInitializeWeights(HIDDEN_LAYER2_SIZE, NUM_LABELS);

    initial_Theta1.astype(np.float32)
    initial_Theta2.astype(np.float32)
    initial_Theta3.astype(np.float32)
 
    args = (INPUT_LAYER_SIZE, HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, NUM_LABELS, X, y, lambda_training)
    init_nn_params = np.hstack([initial_Theta1.flatten(),initial_Theta2.flatten(),initial_Theta3.flatten()]) #Unroll parameters
    res = scipy.optimize.minimize(nn_costfunction2, init_nn_params, args=args,jac = True, method='CG', options={'maxiter':150,'disp':True})

    Theta1 = np.reshape(res.x[0:HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1)], (HIDDEN_LAYER1_SIZE, (INPUT_LAYER_SIZE + 1)))
    Theta2 = np.reshape(res.x[((HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1))):((HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1)))+((HIDDEN_LAYER2_SIZE * (HIDDEN_LAYER1_SIZE + 1)))], (HIDDEN_LAYER2_SIZE, (HIDDEN_LAYER1_SIZE + 1)))
    Theta3 = np.reshape(res.x[((HIDDEN_LAYER1_SIZE * (INPUT_LAYER_SIZE + 1)))+((HIDDEN_LAYER2_SIZE * (HIDDEN_LAYER1_SIZE + 1))):], (NUM_LABELS, (HIDDEN_LAYER2_SIZE + 1)))


    pred = predict(Theta1, Theta2, Theta3, XTest)
        
    pred.astype(np.uint8)
    ytest.astype(np.uint8)

    np.savetxt('salidaEntregar.txt',pred)
    np.savetxt('ytest.txt',ytest)
    
    #accuracy
    print np.mean(pred == ytest.flatten())*100

    # kaggle result

    KaggleData = np.genfromtxt('test.csv', delimiter=',', skip_header=1,dtype=np.uint8)
    XKaggle = KaggleData[:,:]
    
    pred2 = predict(Theta1, Theta2, Theta3, XKaggle)
    pred2.astype(np.uint8)
    np.savetxt('kaggle.txt',pred2)

if __name__ == '__main__':
    main()
