rm(list = ls())

library(readr) # CSV file I/O, e.g. the read_csv function
library(stringr)
library(tm)
library(syuzhet)
library(SnowballC)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

train <-read_csv("../train.csv",n_max =1000)
test <-read_csv("../test.csv",n_max = 1000)