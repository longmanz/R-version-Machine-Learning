#' Implementation of Chapter 2 of Python Machine Learning:
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

# 2.1. a training function of perceptron classifier
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


ML_fit <- function(X, data=NULL){
    # this will take a trained ML_data classifier and a test set, return the predicted outcomes
    if(length(data@type) == 0){
        stop("The input ML_data object has not been trained yet!\n")
    }
    cat(paste("The input object is a ", data@type, "classifier.\n"))
    
    net_input_val = net_input(data@w, as.matrix(X) )
    res <- rep(1, nrow(X))
    res[which(net_input_val < 0)] = -1
    return(res)
}






