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
idx <- sample(1:100, size=80)

training <- iris2[idx, ]
test <- iris2[-idx, ]


############################################
###           begin to train:
############################################

iris_ML <- create_ML_data(X = training[,1:2], y = training$type)

iris_ML_2 <- Perceptron_train(data = iris_ML)

res <- ML_fit(X = test[,1:2], data=iris_ML_2)

sum(res == test[,4])
# a perfect classifier for this dataset.

