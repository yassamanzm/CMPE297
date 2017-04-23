rm(list = ls())

library(readr) # CSV file I/O, e.g. the read_csv function
library(stringr)
library(tm)
library(syuzhet)
library(SnowballC)
library(data.table)
library(h2o)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

h2o.init(nthreads = -1)

ts1 <- fread("../train.csv", select=c("id","question1","question2","is_duplicate"),nrows=1000)

train <-read_csv("../train.csv",n_max =1000)
test <-read_csv("../test.csv",n_max = 1000)

ts1[,":="(question1=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question1),
          question2=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question2))]
ts1[,":="(question1=gsub("  ", " ", question1),
          question2=gsub("  ", " ", question2))]

dt <-data.table(train)
dt[,c(2,3) := NULL]
response <- "is_duplicate"

dt[[response]] <- as.factor(dt[[response]])
predictors <- dt[,c(2,3)]

dt.hex <- as.h2o(dt, destination_frame = "dt.hex", col.types=c("String"))
splits <- h2o.splitFrame(
  data = dt.hex, 
  ratios = c(0.6,0.2),   ## only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex", "test.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

## We only provide the required parameters, everything else is default
## gbm <- h2o.gbm(x = c("question1","question2"), y = "is_duplicate", training_frame = train)
## run h20.gbm on localhost:54321 to see results
## Show a detailed model summary
gbm

## Get the AUC on the validation set
h2o.auc(h2o.performance(gbm, newdata = valid))

