#' Test script for ML methods from Chapter 2. 
#'   We will load the famous Iris dataset from R. 
#'   Or you could download the dataset at:
#'    https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
#'    
#' 
#' Date: 17 July, 2018

library(datasets)
head(iris)

# we will only need two species and two variables 
iris2 <- iris[1:100, c(1, 3, 5)]

# take a look at the dataset. 
plot(iris2[iris2$Species == "setosa", "Sepal.Length"], iris2[iris2$Species == "setosa", "Petal.Length"], 
     col="red", xlim=c(0, 10), ylim=c(0, 10))
points(iris2[iris2$Species != "setosa", "Sepal.Length"], iris2[iris2$Species != "setosa", "Petal.Length"], 
     col="black", xlim=c(0, 10), ylim=c(0, 10))

# create a binary y based on the species
iris2$type <- 1
iris2[iris2$Species != "setosa", "type"] = -1

# Randomly divide the dataset into two part: training set (80 samples) + test set (20 samples)
idx <- sample(1:50, size=40)
idx <- c(idx, idx+50)

training <- iris2[idx, ]
test <- iris2[-idx, ]



############################################
###           begin to train:
############################################

################
##  A. test Perceptron
iris_ML <- create_ML_data(X = training[,1:2], y = training$type)
iris_ML_2 <- Perceptron_train(data = iris_ML)
res <- ML_fit(X = test[,1:2], data=iris_ML_2)
sum(res == test[,4])

################
##  B. test AdalineGD
iris_ML_3 <- AdalineGD_train(data = iris_ML)  # not good. we will need to run standardisation of the variables.
training2 <- training
training2[,1] <- scale(training2[,1])
training2[,2] <- scale(training2[,2])

test2 <- test
test2[,1] <- scale(test2[,1])
test2[,2] <- scale(test2[,2])


iris_ML <- create_ML_data(X = training2[,1:2], y = training2$type)
# the learning rate (eta) is pretty crucial for Adaline. Please try 0.05, 0.01, 0.005, 0.001 and set the n_iter to 50 
#  for better convergence.
iris_ML_3 <- AdalineGD_train(eta = 0.01, n_iter = 50, data = iris_ML)   
res <- ML_fit(X = test2[,1:2], data=iris_ML_3)
sum(res == test2[,4])


################
##  C. test AdalineSGD
iris_ML <- create_ML_data(X = training2[,1:2], y = training2$type)
# the learning rate (eta) is pretty crucial for Adaline. Please try 0.05, 0.01, 0.005, 0.001 and set the n_iter to 50 
#  for better convergence.
iris_ML_4 <- AdalineSGD_train(eta = 0.01, n_iter = 50, data = iris_ML)   
res <- ML_fit(X = test2[,1:2], data=iris_ML_4)
sum(res == test2[,4])

# test the partial fit function of AdalineSGD
iris_ML_5 <- AdalineSGD_train(eta = 0.01, n_iter = 50, data = iris_ML_4, partial_fit = T, add_X = test2[,1:2], add_y = test2[,4])   
res <- ML_fit(X = test2[,1:2], data=iris_ML_5)



################
##  D. test Logistic Regression
iris_ML <- create_ML_data(X = training2[,1:2], y = training2$type)
# the learning rate (eta) is pretty crucial for Adaline. Please try 0.05, 0.01, 0.005, 0.001 and set the n_iter to 50 
#  for better convergence.
iris_ML_6 <- LogisticReg_train(eta = 0.011, n_iter = 50, data = iris_ML)   
res <- ML_fit(X = test2[,1:2], data=iris_ML_6)
sum(res == test2[,4])

# it seems that the average accuracy of LogReg is not very high! There is usually one misspecification.

#' additional note: 
#'  If you are using glm(..., family="binomial"), then it might failed to converge since the iris data
#'   we are using is in 'complete seperation'. This is not an ideal situation for logistic glm() to perform. 











