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

questions <- as.data.table(rbind(ts1[,.(question=question1)], ts1[,.(question=question2)]))
questions <- unique(questions)
questions.hex <- as.h2o(questions, destination_frame = "questions.hex", col.types=c("String"))

STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have", "that",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

tokenize <- function(sentences, stop.words = STOP_WORDS) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
}


dt <-data.table(train)
dt[,c(2,3) := NULL]

dt <- wordStem(dt[,c(2,3)])

if (requireNamespace("mlbench", quietly=TRUE)) {

  df <- as.h2o(dt)
  rng <- h2o.runif(df, seed=1234)
  train <- df[rng<0.8,]
  valid <- df[rng>=0.8,]
  plot.H2OTabulate
  h2o.randomForest(dt[,c(2,3)], dt[,c(4)], dt[,c(1)], model_id = NULL,
                   validation_frame = NULL, nfolds = 3,
                   keep_cross_validation_predictions = FALSE,
                   keep_cross_validation_fold_assignment = FALSE,
                   score_each_iteration = FALSE, score_tree_interval = 0,
                   fold_assignment = c("AUTO", "Random", "Modulo", "Stratified"),
                   fold_column = NULL, ignore_const_cols = TRUE, offset_column = NULL,
                   weights_column = NULL, balance_classes = FALSE,
                   class_sampling_factors = NULL, max_after_balance_size = 5,
                   max_hit_ratio_k = 0, ntrees = 50, max_depth = 20, min_rows = 1,
                   nbins = 20, nbins_top_level = 1024, nbins_cats = 1024,
                   r2_stopping = Inf, stopping_rounds = 0, stopping_metric = c("AUTO",
                                                                               "deviance", "logloss", "MSE", "RMSE", "MAE", "RMSLE", "AUC", "lift_top_group",
                                                                               "misclassification", "mean_per_class_error"), stopping_tolerance = 0.001,
                   max_runtime_secs = 0, seed = -1, build_tree_one_node = FALSE,
                   mtries = -1, sample_rate = 0.6320000291, sample_rate_per_class = NULL,
                   binomial_double_trees = FALSE, checkpoint = NULL,
                   col_sample_rate_change_per_level = 1, col_sample_rate_per_tree = 1,
                   min_split_improvement = 1e-05, histogram_type = c("AUTO",
                                                                     "UniformAdaptive", "Random", "QuantilesGlobal", "RoundRobin"),
                   categorical_encoding = c("AUTO", "Enum", "OneHotInternal", "OneHotExplicit",
                                            "Binary", "Eigen"))
  plot(randomforest)
  plot(randomforest, timestep = "duration", metric = "deviance")
  plot(randomforest, timestep = "number_of_trees", metric = "deviance")
  plot(randomforest, timestep = "number_of_trees", metric = "rmse")
  plot(randomforest, timestep = "number_of_trees", metric = "mae")
}