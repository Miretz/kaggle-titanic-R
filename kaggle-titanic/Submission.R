rm(list=ls())

set.seed(333)

library(rpart)
library(rpart.plot)
library(randomForest)
library(party)
library(caret) 
library(stringi)
library(stringr)
library(parallel)
library(doParallel)
library(dplyr)
library(mice)

#
# Create the title column
#
createTitleColumn <- function(dataset) {
  dataset$Title <- gsub('(.*, )|(\\..*)', '', dataset$Name)
  # Titles with very low cell counts to be combined to "rare" level
  rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                  'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
  
  # Also reassign mlle, ms, and mme accordingly
  dataset$Title[dataset$Title == 'Mlle']        <- 'Miss' 
  dataset$Title[dataset$Title == 'Ms']          <- 'Miss'
  dataset$Title[dataset$Title == 'Mme']         <- 'Mrs' 
  dataset$Title[dataset$Title %in% rare_title]  <- 'Rare Title'
  dataset$Title <- as.factor(dataset$Title)
  return(dataset)
}


#
# split function
#
splitFunction <- function(x) {
  return(strsplit(x, split='[,.]')[[1]][1])
}


#
# Create FamilyId title
#
createFamilyIds <- function(dataset){
  dataset$FamilySize <- dataset$SibSp + dataset$Parch + 1 # Engineer FamilySize
  
  # Engineered variable: FamilyID
  name <- as.character(dataset$Name)
  dataset$Surname <- sapply(name, FUN=splitFunction)
  
  dataset$FamilyID <- paste(as.character(dataset$FamilySize), dataset$Surname, sep="")
  dataset$FamilyID[dataset$FamilySize <= 4] <- 'Small'
  famIDs <- data.frame(table(dataset$FamilyID))
  famIDs <- famIDs[famIDs$Freq <= 4,]
  dataset$FamilyID[dataset$FamilyID %in% famIDs$Var1] <- 'Small'
  dataset$FamilyID <- as.factor(dataset$FamilyID)
  
  # Family Size Bucket
  dataset$FamilySizeBucket[dataset$FamilySize == 1] <- 'singleton'
  dataset$FamilySizeBucket[dataset$FamilySize <= 4 & dataset$FamilySize > 1] <- 'small'
  dataset$FamilySizeBucket[dataset$FamilySize > 4] <- 'large'
  dataset$FamilySizeBucket <- as.factor(dataset$FamilySizeBucket)
  
  return(dataset)
}


#
# Predict missing Fare
#
predictMissingFareValues <- function(dataset){
  dataset$Fare[1044] <- median(dataset[dataset$Pclass == '3' & dataset$Embarked == 'S', ]$Fare, na.rm = TRUE)
  # dataset$Fare[1044] <- median(dataset$Fare, na.rm=TRUE)
  
  dataset$FareBucket <- cut(dataset$Fare, 4)
  
  return(dataset)
}

#
# Predict missing Age
#
predictMissingAgeValues <- function(dataset){
  # Make variables factors into factors
  factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                   'Title','Surname','FamilyID','FamilySizeBucket')
  
  dataset[factor_vars] <- lapply(dataset[factor_vars], function(x) as.factor(x))
  mice_mod <- mice(
    dataset[, !names(dataset) %in%
              c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf')
  mice_output <- complete(mice_mod)
  dataset$Age <- mice_output$Age
  
  dataset$Child <- 'Child'
  dataset$Child[dataset$Age >= 18] <- 'Adult'
  dataset$Child <- as.factor(dataset$Child)
  
  dataset$Mother <- 'Not Mother'
  dataset$Mother[dataset$Sex == 'female' & 
                   dataset$Parch > 0 & dataset$Age > 18 & dataset$Title != 'Miss'] <- 'Mother'
  dataset$Mother <- as.factor(dataset$Mother)
  
  dataset$AgeBucket <- cut(dataset$Age, 5)
  
  return(dataset)
}


#
# Parse Cabin numbers
#
parseCabinNumbers <- function(dataset){
  # Cabin
  cabin <- as.character(dataset$Cabin)
  countSpaces <- function(s) { sapply(gregexpr(" ", s), function(p) { sum(p>=0) } ) }
  dataset$Cabins <- sapply(cabin, FUN=countSpaces) + 1 
  dataset$Cabins[cabin == ""] <- 0
  
  dataset$Deck <- sapply(cabin, FUN = function(p){substr(p, 1, 1)})
  dataset$Deck <- as.factor(dataset$Deck)
  
  dataset$Ticket <- as.character(dataset$Ticket)
  dataset$DeckTkt <- sapply(dataset$Ticket, FUN = function(p){substr(p, 1, 1)})
  dataset$DeckTkt <- as.factor(dataset$DeckTkt)
  
  dataset$Ticket <- as.factor(dataset$Ticket)
  dataset <- transform(dataset, TicketShare = table(Ticket)[Ticket])
  
  return(dataset)
}



#
# Preprocess data set
#
preprocessDataset <- function(dataset){
  dataset$Embarked <- as.character(dataset$Embarked)
  dataset$Embarked[c(62,830)] = "S" # Fill embarked
  dataset$Embarked <- as.factor(dataset$Embarked)
  
  dataset <- createTitleColumn(dataset)
  dataset <- createFamilyIds(dataset)
  dataset <- predictMissingFareValues(dataset)
  dataset <- predictMissingAgeValues(dataset)
  dataset <- parseCabinNumbers(dataset)
  
  #factorization
  dataset$Pclass <- as.factor(dataset$Pclass)
  dataset$Sex <- as.factor(dataset$Sex)
  dataset$Cabins <- as.factor(dataset$Cabins)
  dataset$FamilySize <- as.factor(dataset$FamilySize)
  
  return(dataset)
}


# Join together the test and train sets for easier feature engineering
# import train and test datasets
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

test$Survived <- NA
combi <- rbind(train, test)
combi <- preprocessDataset(combi)
# str(combi)

# Split back into test and train sets
train <- combi[1:891,]
test <- combi[892:1309,]

# crossvalidation set
# train test split
train_ind <- createDataPartition(y = train$Survived, p= 0.7, list = FALSE)
train_1 <- train[train_ind, ]
valid_1 <- train[-train_ind, ]

# str(train)
# combi[!complete.cases(combi), ]
# sapply(train, anyNA) # same as above
# any(duplicated(train)) # check duplicated rows


#-------------------------------------------------------
# MODEL
#-------------------------------------------------------

grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75))

trctrl <- trainControl(allowParallel = TRUE, method = 'repeatedcv', 
                       number = 10, repeats = 3)

# enable parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)


system.time({
  fit <- train(
    as.factor(Survived) ~
      Pclass + Sex + Title + AgeBucket + FamilySizeBucket + Mother + FareBucket + FamilyID + Deck + Cabins + DeckTkt + TicketShare.Freq,
    data = train_1,
    method = "svmLinear",
    trControl = trctrl,
    tuneGrid = grid,
    tuneLength = 10
  )
})
fit

# stop the cluster
stopCluster(cluster)
registerDoSEQ()

# cross validation
Prediction <- predict(fit, valid_1)
results <- table(Prediction, valid_1$Survived)
confusionMatrix(results)

# Variable importances
varImp(fit)

# submission
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "result.csv", row.names = FALSE)
