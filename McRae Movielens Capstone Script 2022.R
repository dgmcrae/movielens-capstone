## McRae Movielens Capstone Project R Script

##########################################################
# First reate edx set, validation set (final hold-out test set)
# using supplied code
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosytem", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(recosystem)
library(kableExtra)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Create probe set for model testing with 10 % of edx data, 
## and a new train set, edx_train, with the remaining 90% of edx
set.seed(2, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(2)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in probe set are also in edx_train set
edx_probe <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from probe set back into edx_train set
removed <- anti_join(temp, edx_probe)
edx_train <- rbind(edx_train, removed)

rm(test_index, temp, removed)

## We now use recosystem to tune a matrix factorisation model. We first set.seed()
## as recosystem uses a randomised algorithm
set.seed(3, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(3)`

# recommenderlab uses three column data object ("sparse matrix triplet form') 
# of userId, movieId and rating for train set, and just the userId and movieId for test set. 
# We create these from edx_train and edx_probe respectively, as well as creating
# a model object r
train_set = data_memory(edx_train$userId, edx_train$movieId, rating = edx_train$rating)
test_set  = data_memory(edx_probe$userId, edx_probe$movieId, rating = NULL)
r = Reco()

# We now tune the model on train_set using recosystem's tune function, 
# using five fold cross validation. I specify user-defined ranges of parameters
# for the dimensions and learning rate for gradient descent, but use recosystem's 
# default ranges for the four regularization parameters. Full details in accompanying
# report. nthread sets parallelization which significantly speeds up this step, 
# but I have set nthread to 1 to keep results fully reproducible. Using parallelisation 
# still converges to the same tuned parameters (eg nthread = 4 for my four core processor), 
# but appears not to produce a reproducible loss function error, perhaps because of 
# randomisation in the underlying LIBMF library (see report). 

# NB: Using nthread = 1, tuning may take several hours, so I have provided two versions 
# of the code below - one that requires the user to run the tuning code, and one that
# manually inserts the tuned parameters which enables the user to skip tuning opts

opts = r$tune(train_set, opts = list(dim = c(10, 20, 30, 40, 50), lrate = c(0.1, 0.2),
                                      nthread = 1, niter = 10))

# recosystem stores tuned parameters as opts$min
opts$min

## We now use recosystem's train function, again with parallel processing set to 1 
## to keep results reproducible. As we can specify the number of training iterations,
## we use sapply to test 10,20,30, 40 and 50 iterations to find the optimal number. Use
## appropriate version of code to define errors, depending on whether or not you
## tuned parameters

# No of iterations for sapply to iterate over
iters <- seq(10,50,10)

# Vector to store probe set RMSE for different no of iterations
errors <- vector(mode = "numeric", length = 5)

# If you tuned parameters above, use this version of user-defined function 
# to generate RMSEs for each no of iterations
errors <- sapply(iters, function(x){
  r$train(train_set, opts = c(opts$min,
                              nthread = 1, niter = x))
  
  pred = r$predict(test_set, out_memory())
  RMSE(edx_probe$rating, pred)
})

# If you skipped parameter tuning, tuned parameters are entered manually here
errors <- sapply(iters, function(x){
  r$train(train_set, opts = c(dim = 50, costp_l1 = 0,
                              costp_l2 = 0.01, costq_l1 = 0,
                              costq_l2 = 0.1, lrate = 0.1,
                              nthread = 1, niter = x))
  
  pred = r$predict(test_set, out_memory())
  RMSE(edx_probe$rating, pred)
})

# Plot RMSE vs iterations to identify optimal iterations
# (20 iterations minimises the RMSE)
tibble(iters, errors) %>% ggplot(aes(iters, errors)) + geom_line() + 
  labs(x = "No of iterations", y = "RMSE on probe set") +
  ggtitle("Probe set RMSE per training iterations")

# We now train the model on edx_train using 20 iterations - again
# please use the appropriate version of the code

# If you tuned parameters
r$train(train_set, opts = c(opts$min,
                            nthread = 1, niter = 20))

# If you skipped parameter tuning
r$train(train_set, opts = c(dim = 50, costp_l1 = 0,
                            costp_l2 = 0.01, costq_l1 = 0,
                            costq_l2 = 0.1, lrate = 0.1,
                            nthread = 1, niter = 20))

## Now that our model is trained, we create a vector of predictions
## from the test_set we defined as userId and movieId of edx_probe
## ('out_memory()' saves vector as a data object instead of a file)
pred_rvec = r$predict(test_set, out_memory())

# Use supplied code to set up a function 
# to calculate root mean squared error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# We calculate RMSE on our probe set
RMSE(edx_probe$rating, pred_rvec)

## As the RMSE easily satisifies success threshold for challenge,
## we now train our model on all of edx using the parameters we 
## tuned on edx_train and then test the model on our hold-out test set
set.seed(3, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(3)`

# Define new model object, and new test and train sets from edx and validation 
rfinal = Reco()
final_train = data_memory(edx$userId, edx$movieId, rating = edx$rating)
final_test = data_memory(validation$userId, validation$movieId, rating = NULL)

# Train the model on all of edx - again two versions of code provided

# If you tuned parameters above
rfinal$train(final_train, opts = c(opts$min, nthread = 1, niter = 20))

# If you skipped parameter tuning
rfinal$train(final_train, opts = c(dim = 50, costp_l1 = 0,
                                   costp_l2 = 0.01, costq_l1 = 0,
                                   costq_l2 = 0.1, lrate = 0.1, nthread = 1, niter = 20))

# Create a vector or predictions from the userId and movieId columns of validation
pred_rvec_final = rfinal$predict(final_test, out_memory())

# Final testing of the model on our hold-out test set, validation
final_rmse <- RMSE(validation$rating, pred_rvec_final)

#####################################################################################
## The prediction task is completed above - the following is the code for the figures
## and summary stats in the report (also available in the Rmd file)
#####################################################################################

# Figure 1 - plot of distribution of ratings
edx %>% group_by(rating) %>% summarise(n = n()) %>% 
  mutate(inte = ifelse(rating == round(rating), 1, 0)) %>% 
  ggplot(aes(rating, n, fill = inte)) + geom_col(show.legend = FALSE) +
  labs(x = "Movie rating", y = "No of ratings") + 
  scale_y_continuous(breaks = seq(0,3000000, 250000)) +
  scale_x_continuous(breaks  = seq(0,5, 0.5)) + 
  ggtitle("Figure 1: No of each possible rating in the train set") +
  theme(plot.title=element_text(size=12))

# No distinct movies in provided train set
edx %>% distinct(movieId, .keep_all = TRUE) %>% nrow()

# Comparison of sparsity - matrix of 1000 most active users and movies
# and matrix of 1000 least active movies and users

top_1K_users <- edx %>% count(userId, sort = TRUE) %>% slice_head(., n = 1000) %>% pull(userId)
top_1K_movies <- edx %>% count(movieId, sort = TRUE) %>% slice_head(., n = 1000) %>% pull(movieId)

((edx %>% filter(userId %in% top_1K_users & movieId %in% top_1K_movies) %>% nrow())*100)/
  (length(top_1K_users)*length(top_1K_movies)) #percentage of cells containing ratings
  
bottom_1K_users <- edx %>% count(userId) %>% arrange(n) %>% slice_head(., n = 1000) %>% pull(userId)
bottom_1K_movies <- edx %>% count(movieId) %>% arrange(n) %>% slice_head(., n = 1000) %>% pull(movieId)

# Code for table of sample lines from dataset
edx[c(12345, 123456, 1234567),] %>%  kbl(caption = "Sample entries, train set") %>% 
  kable_paper("hover", full_width = F) %>% kable_styling(latex_options = "HOLD_position")

# Figure 2 - plot of average rating per movie against number of reviews per movie, with a trend line
edx %>% group_by(movieId, title) %>% summarise(ratings = mean(rating), no_reviews = n()) %>% 
  ggplot(aes(no_reviews, ratings)) + geom_point(shape = '.') + 
  geom_smooth() + 
  scale_x_log10() + labs(x = "No of reviews per movie (log scale)", y = "Average rating") + 
  ggtitle("Figure 2: Average rating vs times reviewed, per movie") +
  theme(plot.title=element_text(size=12))

# Figure 3 - plot of average rating per user against number of reviews per user, with a trend line
edx %>% group_by(userId) %>% summarise(ratings = mean(rating), no_reviews = n()) %>% 
  ggplot(aes(no_reviews, ratings)) + geom_point(shape = '.') + 
  geom_smooth() + 
  scale_x_log10() + labs(x = "No of reviews per user (log scale)", y = "Average rating") +
  ggtitle("Figure 3: Average rating vs no of reviews, per user") +
  theme(plot.title=element_text(size=12))

## Irizarry machine learning predictions
## applied to edx_train and edx_probe
# First, just the mean rating
mu <- mean(edx_train$rating)
naive_rmse <- RMSE(edx_probe$rating, mu)

# Next, adding a movie bias term
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edx_probe %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, edx_probe$rating)

#adding a user bias term
user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings1 <- edx_probe %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings1, edx_probe$rating)

# Generating a table of ratings with each of Irizarry's simple methods
rmse_results <- tibble(Method = c("Just the average", "Movie effects model", "Movie + user effects model"),
                       RMSE = c(naive_rmse, model_1_rmse, model_2_rmse))

rmse_results %>%  kbl(caption = "Performance of Irizarry's simple prediction models") %>% 
  kable_paper("hover", full_width = F) %>% kable_styling(latex_options = "HOLD_position")

# Code to generate table 3 - tuning parameters
params <- tibble(Parameter = c("dim", "costp_l1", "costp_l2", "costq_l1", "costq_l2", "lrate"),
                 Description = c("Number of latent factors", 
                                 "L1 regularization cost for user factors", 
                                 "L2 regularization cost for user factors", 
                                 "L1 regularization cost for item factors", 
                                 "L2 regularization cost for item factors", 
                                 "Learning rate/step size in gradient descent"),
                 Status = c("User-defined", "Default", "Default", "Default", "Default", "User-defined"),
                 Range = c("10, 20, 30, 40, 50", "0, 0.1", "0.01, 0.1", "0, 0.1", "0.01, 0.1", "0.1, 0.2"),
                 Tuned_Parameter = c("50", 0, 0.01, 0, 0.1, 0.1))

params %>% kbl(caption = "Parameters used for tuning and final tuned settings", 
               col.names = c("Parameter", "Description", "Status", "Range", "Tuned Parameter")) %>%  
  kable_paper("hover", full_width = F) %>% kable_styling(latex_options = "HOLD_position")

# If you ran code to generate opts above, max and min for squared error 
# during parameter tuning are as follows
max(opts[["res"]][["loss_fun"]])
min(opts[["res"]][["loss_fun"]])

## Code for Figure 4 - plot of Probe set RMSE per training iteration
## is already provided above as a step in choosing the number of
## iterations to use to train the model on edx. Only the tile of the 
## plot is altered in the RMD file for the report - i.e. the plot is
## labelled Figure 4.


