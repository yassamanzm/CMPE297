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
questions <- as.data.table(rbind(ts1[,.(question=question1)], ts1[,.(question=question2)]))
questions <- unique(questions)
questions.hex <- as.h2o(questions, destination_frame = "questions.hex", col.types=c("String"))
is_duplicate <- as.data.table(ts1[,.(duplicate=is_duplicate)])
is_duplicate.hex <- as.h2o(is_duplicate, destination_frame = "duplicate.hex", col.types=c("Number"))

STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

tokenize <- function(sentences, stop.words = STOP_WORDS) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  
  # convert to lower case
  tokenized.lower <- h2o.tolower(tokenized)
  # remove short words (less than 2 characters)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]
  # remove stop words
  tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% STOP_WORDS),]
}

predict <- function(questions.hex, w2v, gbm) {
  words <- tokenize(as.character(as.h2o(questions.hex)))
  questions.vec <- h2o.transform(w2v, words, aggregate_method = "AVERAGE")
  h2o.predict(gbm, questions.vec)
}

print("Break job titles into sequence of words")
words <- tokenize(questions)

print("Build word2vec model")
w2v.model <- h2o.word2vec(words, sent_sample_rate = 0, epochs = 10)

print("Sanity check - find synonyms for the word 'love'")
print(h2o.findSynonyms(w2v.model, "love", count = 5))

print("Calculate a vector for each job title")
questions.vecs <- h2o.transform(w2v.model, words, aggregate_method = "AVERAGE")

print("Prepare training&validation data (keep only job titles made of known words)")
valid.questions <- ! is.na(questions.vecs$C1)
data <- h2o.cbind(job.titles[valid.questions, "category"], questions.vecs[valid.questions, ])
data.split <- h2o.splitFrame(data, ratios = 0.8)

print("Build a basic GBM model")
gbm.model <- h2o.gbm(x = names(questions.vecs), y = "category",
                     training_frame = data.split[[1]], validation_frame = data.split[[2]])


if (requireNamespace("mlbench", quietly=TRUE)) {
  
  df <- as.h2o(dt)
  rng <- h2o.runif(df, seed=1234)
  train <- df[rng<0.8,]
  valid <- df[rng>=0.8,]
  plot.H2OTabulate
  
  plot(randomforest)
}

