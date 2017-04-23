rm(list = ls())

library(readr) # CSV file I/O, e.g. the read_csv function
library(stringr)
library(tm)
library(syuzhet)
library(SnowballC)
library(data.table)
library(h2o)
library(neuralnet)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

h2o.init(nthreads = -1)

ts1 <- fread("../train.csv", select=c("id","question1","question2","is_duplicate"),nrows=1000)

train <-read_csv("../train.csv",n_max =2000)
test <-read_csv("../test.csv",n_max = 2000)

ts1[,":="(question1=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question1),
          question2=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question2))]
ts1[,":="(question1=gsub("  ", " ", question1),
          question2=gsub("  ", " ", question2))]

dt <-data.table(train)

dt[,":="(question1=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question1),
          question2=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question2))]
dt[,":="(question1=gsub("  ", " ", question1),
          question2=gsub("  ", " ", question2))]
dt[,c(2,3) := NULL]
# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

# Date: 03/19/2017  
# Proj: Quora Kaggle Competetion 
# Name: Nachiket Garge

rm(list = ls())


#library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(stringr)
library(tm)
library(syuzhet)
library(SnowballC)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

train <-read_csv("../input/train.csv",n_max =1000)
test <-read_csv("../input/test.csv",n_max = 1000)
#dim(train)

# For each question preparing a score with same important words
# For eg. If we consider question8
# train$question1[8]
# "How can I be a good geologist?"
# train$question2[8]
# "What should I do to be a great geologist?"
# After data filtering, stemming, removing puctuations, removing stop words and converting all words to lower case
# we have following results,
# > a11
#[1] "can"       "geologist" "good"     
#> b11
#[1] "geologist" "great"
# Now calculating score with common words
# score = 2/5 (as there are 2 common words within total of 5 words)
# We can predict the target with help of this score

# This method may lead to mis interpretation in case of opposite sentences 
# Thus,
# Through sentiment analysis we can provide positive and negative emotions as predictors to improve accuracy

# With these regressors I achieved the accuracy of 69% 

# Please find the code below:

df = data.frame()
df.new = data.frame()

text = function(train){
  
  for (i in 1:nrow(train))
  {
    # cbind(train,adist = diag(adist(train$question1,train$question2)))
    q1 <- Corpus(VectorSource(train$question1[i]))
    # Performing the Text cleaning
    #Removing punctuations
    q1 <- tm_map(q1, removePunctuation)   
    #inspect(q1)
    # To remove punctuations on the text data
    #Removing Numbers removeNumbers
    q1 <- tm_map(q1, removeNumbers)   
    #inspect(q1)
    #To avoid duplicacy converting all to lower case tolower
    q1 <- tm_map(q1, tolower)   
    #inspect(q1)
    # removing common word endings stemDocument
    q1 <- tm_map(q1, stemDocument)   
    #inspect(q1)
    # to remove white space stripWhitespace
    q1 <- tm_map(q1, stripWhitespace)   
    #inspect(q1)
    #Removing Stop Words as they don't add any value
    #Smart is inbuilt, which gives a set of pre-defined stop words.  removeWords, stopwords("english")
    q1 <- tm_map(q1, removeWords, stopwords("english"))   
    #inspect(q1)
    # to convert documents into text documents...this tells R to treat preprocessed document as text documents PlainTextDocument
    q1 <- tm_map(q1, PlainTextDocument)   
    #inspect(q1)
    doc = TermDocumentMatrix(q1) 
    a11 = doc$dimnames$Terms
    
    q2 <- Corpus(VectorSource(train$question2[i]))
    # Performing the Text cleaning
    #Removing punctuations
    q2 <- tm_map(q2, removePunctuation)   
    # To remove punctuations on the text data
    #Removing Numbers removeNumbers
    q2 <- tm_map(q2, removeNumbers)   
    #To avoid duplicacy converting all to lower case tolower
    q2 <- tm_map(q2, tolower)   
    # removing common word endings stemDocument
    q2 <- tm_map(q2, stemDocument)   
    # to remove white space stripWhitespace
    q2 <- tm_map(q2, stripWhitespace)   
    #Removing Stop Words as they don't add any value
    #Smart is inbuilt, which gives a set of pre-defined stop words.  removeWords, stopwords("english")
    q2 <- tm_map(q2, removeWords, stopwords("english"))   
    # to convert documents into text documents...this tells R to treat preprocessed document as text documents PlainTextDocument
    q2 <- tm_map(q2, PlainTextDocument)   
}



response <- "is_duplicate"

dt[[response]] <- as.factor(dt[[response]])
predictors <- dt[,c(2,3)]

dt.hex <- as.h2o(dt, destination_frame = "dt.hex", col.types=c("String","Number","Enum"))
splits <- h2o.splitFrame(
  data = dt.hex, 
  ratios = c(0.6,0.2),   ## only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex", "test.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]
dtmatrix <- model.matrix(~dt$question1+dt$question2+dt$is_duplicate,data=dt)

train.data <- data.frame(as.matrix(dt))
##neuralnet(names(train[,1]~names(train[,c(2)]+names(train[,c(3)]), train.data,
  ##        err.fct="ce", linear.output=FALSE, likelihood=TRUE)))
mode(dt)<-"numeric"
neuralnet(dt$is_duplicate~as.numeric(dt$question1)+as.numeric(dt$question2), dt,
                                err.fct="sse", linear.output=FALSE, likelihood=TRUE)
## We only provide the required parameters, everything else is default
## gbm <- h2o.gbm(x = c("question1","question2"), y = "is_duplicate", training_frame = train)
## run h20.gbm on localhost:54321 to see results
## Show a detailed model summary
## train.h2o$supplier = as.factor(train.h2o$supplier)

