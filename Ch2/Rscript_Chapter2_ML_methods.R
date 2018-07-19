#' Implementation of Chapter 2 of Python Machine Learning (PML):
#'    1. Perceptron() : Perceptron classifier
#'    2. AdalineGD()  : Adaptive linear neuron classifier with gradient descent
#'    3. AdalineSGD() : Adaptive linear neuron classifier with Stochastic gradient descent
#'    
#' Version: 1.0
#' Author: Longda
#' Date: 17 July, 2018


##################################################
###     1. ML_data class
##################################################

# 1.1. create a S4 object called ML_data, which contains 4 elements:
#    1. X is the X matrix
#    2. y is the y vector (the true outcome)
#    3. w is the weights.
#    4. errors: is a vector to store the error terms from each epoch
setClass("ML_data", slots=list(X="matrix", y="numeric", w="numeric", errors="numeric", type="character"))  

# set an $ operator for this class
setMethod("$", "ML_data",
          function(x, name)
          {
              ## 'name' is a character(1)
              slot(x, name)
          })

# 1.2. a function that help user create a ML_data object
create_ML_data <- function(X = NULL, y = NULL){
    return ( new("ML_data", X=as.matrix(X), y=as.vector(y)) )
}



##################################################
###     2. Perceptron classifier
##################################################

# 2.0.  Create a function to calculate the net_input (wX)
net_input <- function(w, X){
    return (X%*%w[-1] + w[1])
}

sigmoid <- function(z){
    return (1/(1+exp(-z)))
}

# 2.1. a perceptron classifier
#  This is a perceptron classifier, which will take in a training set and train 
#   for a classifier (weights).
# 
# 1. eta: the learning rate
# 2. n_iter: number of maximum iteration
# 3. data: a ML_data object that contains X, y, and w. 
#

Perceptron_train <- function(eta = 0.01, n_iter = 10, data = NULL){
    # set the ML_data object's type to perceptron classifier
    data@type <- "Perceptron"
    # initialise the weights
    data@w <- rep(0, ncol(data@X)+1)
    
    # run iteration
    for(i in 1:n_iter){
        error = 0
        for(j in 1:nrow(data@X)){
            net_input_val = net_input(data@w, data@X[j,])
            res = ifelse(net_input_val >= 0, 1, -1)
            if(res != data@y[j]){
                update = eta*(data@y[j] - res)
                data@w[-1] = data@w[-1] + update*data@X[j,]
                data@w[1] = data@w[1] + update
                error = error + 1
            }
        }
        data@errors = c(data@errors, error)
    }
    return (data)
}



# 2.2. a Adaptive Linear Neuron (Adaline) classifier, version 1. 
#  This is an Adaline classifier with Gradient Descent, which will take in a training set and train 
#   for a classifier (weights).
# 
# 1. eta: the learning rate
# 2. n_iter: number of maximum iteration
# 3. data: a ML_data object that contains X, y, and w. 

AdalineGD_train <- function(eta = 0.01, n_iter = 10, data = NULL){
    # set the ML_data object's type to perceptron classifier
    data@type <- "AdalineGD"
    # initialise the weights
    data@w <- rep(0, ncol(data@X)+1)
    
    # run iteration
    for(i in 1:n_iter){
        net_input_val = net_input(data@w, data@X)
        error = data@y - net_input_val
        cost = sum((error)^2)*0.5
        update = eta*t(data@X) %*% error
        data@w[-1] = data@w[-1] + update
        data@w[1] = data@w[1] + eta*sum(error)

        data@errors = c(data@errors, cost)
    }
    return (data)
}



# 2.3. a Adaline classifier, version 2. 
#  This is an Adaline classifier with Stochastic Gradient Descent. 
#  

AdalineSGD_train <- function(eta = 0.01, n_iter = 10, data = NULL, partial_fit = F, shuffle = T, add_X = NULL, add_y = NULL){
    # set the ML_data object's type to perceptron classifier
    data@type <- "AdalineSGD"
    # initialise the weights 
    if (partial_fit == F | length(data@w) == 0){
        data@w <- rep(0, ncol(data@X)+1)
    }
    
    # Part A. Run iteration from the scratch, i.e. partial_fit == F
    if(partial_fit == F){
        for(i in 1:n_iter){
            if(shuffle == T){
                idx = sample(1:nrow(data@X), size=nrow(data@X))
                X = data@X[idx,]
                y = data@y[idx]
            }
            costs = vector()
            for(j in 1:nrow(X)){
                net_input_val = net_input(data@w, X[j,])
                error = y[j] - net_input_val
                cost = (error)^2*0.5
                update = eta*(X[j,] %*% error)
                data@w[-1] = data@w[-1] + update
                data@w[1] = data@w[1] + eta*error
                costs = c(costs, cost)
            }
            
            data@errors = c(data@errors, mean(costs) )
        }
    } else {
        # Part B. 
        # the partial fit function for on-line learning, i.e. the previous data@w is to be kept. 
        add_X <- as.matrix(add_X)
        add_y <- as.matrix(add_y)
        if(shuffle == T){
            idx = sample(1:nrow(add_X), size=nrow(add_X))
            X = add_X[idx,]
            y = add_y[idx]
        }
        costs = vector()
        for(j in 1:nrow(X)){
            net_input_val = net_input(data@w, X[j,])
            error = y[j] - net_input_val
            cost = (error)^2*0.5
            update = eta*(X[j,] %*% error)
            data@w[-1] = data@w[-1] + update
            data@w[1] = data@w[1] + eta*error
            costs = c(costs, cost)
        }
        
        data@errors = c(data@errors, mean(costs) )
    }
    
    
    return (data)
}




# 2.4. Logistic regression classifier. 
#  This is an logistic regression classifier, which is very similar the the Adaline. 
#   The only difference is for Adaline, the activation function is just an identity function, 
#    but for logistic regression, the activation function is a "sigmoid function". 
# 
#  Note that the final fit function also requires a different Quantizer.  

LogisticReg_train <- function(eta = 0.01, n_iter = 10, data = NULL){
    # set the ML_data object's type to logistic regression classifier
    data@type <- "Logistic regression"
    # initialise the weights
    data@w <- rep(0, ncol(data@X)+1)
    
    # notice in LogReg the target should be coded as 0 and 1, instead of -1 and 1. 
    # we need to recode it 
    y = ifelse(data@y == 1, 1, 0)
    
    # run iteration
    for(i in 1:n_iter){
        net_input_val = net_input(data@w, data@X)
        sigmoid_val = sigmoid(net_input_val)    # remember to do this sigmoid transformation
        
        # the cost function is a bit weird, please refer to page 87 of PML,
        # we add a very small values 1e-32 to avoid log(0) and produces NaN
        cost = -sum(y*log(sigmoid_val + 1e-32)) - sum((1-y)*log(1-sigmoid_val + 1e-32))            

        inter = ifelse(y == 1, 1/sigmoid_val, -1/(1-sigmoid_val) )
        update = eta*t(inter)%*%data@X
        
        data@w[-1] = data@w[-1] + update
        data@w[1] = data@w[1] + eta*sum(inter)
        
        data@errors = c(data@errors, cost)
    }
    return (data)
}





##################################################
###     3. Predict funtion
##################################################

# 3.1. This is a predict function. 
#  It takes a test set and a trained classifier (a trained ML_data object)
#   and returns the predicted values.

ML_fit <- function(X, data=NULL){
    if(length(data@type) == 0){
        stop("The input ML_data object has not been trained yet!\n")
    }
    cat(paste("The input object is a ", data@type, "classifier.\n"))
    
    net_input_val = net_input(data@w, as.matrix(X) )
    
    if (data@type == "Logistic regression"){
        # the logistic regression requires a different activation function (the sigmoid function)
        inter = sigmoid(net_input_val)
        res <- ifelse(inter >= 0.5, 1, -1)
    } else{
        res <- ifelse(net_input_val < 0, -1, 1)
    }
    return(res)
}






