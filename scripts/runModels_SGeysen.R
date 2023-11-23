#      Run subject level models -- Version 1
# Last edit:    2023/11/23
# Author:       Geysen, Steven (SG)
# Notes:        - Run the different subject level models for the task used by
#                   Marzecova et al. (2019)
#               - Release notes:
#                   * Initial commit
# To do:        - Create models
#               - Run models
# Comments:     SG: There is no need to set the working directory when using
#                   the project. Without project prefix all paths with '../'.
# Sources:      https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
# --------------



#### Imports and directories ####


# Clear working memory
rm(list=ls())


# Libraries
library(dplyr)
library(loo)
library(reshape2)
library(rstan)
library(shinystan)
## Stan settings
rstan_options(auto_write=TRUE)


# Load data
undata <- read.csv('data/UnDataProjectSteven.csv')
