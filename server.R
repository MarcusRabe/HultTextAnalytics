######################################################################################
# Playing around with some Analysis on the data captured during the
# Hult MBAN Text Analytics Class (Feb 2019)
######################################################################################
# Please keep in mind that the data was captured usig MS office dictation which didn't
# work well. This results in a very messy dataset. Another limitation is the very small
# number of responses in the dataset.
######################################################################################
# The survey the data is based on was designed to analyse the brand sentiment for
# two brands of sneakers and asked the following questions:
# Q1: What do you look for in a sneaker?
# Q2: What do you like to do in your free time?
# Q3: How old are you?
# Q4: What is your Gender?
# Q5: Do you prefer Nike or Addidas?
######################################################################################
# In order to run the H2O library install the H2O java library first:
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html
# Don't omit the java installation (steps 1 and 2) as the R package won't run without
######################################################################################
# Author: Marcus Rabe and Team 12
######################################################################################

library(shiny)
library(tidyverse)
library(tidytext)
library(dplyr)
library(readr)
library(Matrix)
library(wordcloud)
library(ggplot2)
library(reshape2)
library(h2o)

## create dashboard

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  # initialise H2O ML
  h2o.init()
  
  # read data (based on current dir)
  bc <- read_csv(paste(dirname(rstudioapi::getSourceEditorContext()$path),
                       "/resp_1 .csv",
                       sep = ""))
  answers = h2o.importFile(paste(dirname(rstudioapi::getSourceEditorContext()$path),
                                 "/Data2.csv",
                                 sep = ""),
                           header = TRUE,
                           col.names = c("LookFor",
                                         "FreeT",
                                         "Age",
                                         "Gender",
                                         "AvsN"))
  
  #######
  # run the ML
  #######
  
  num_train = 45 # records in training dataset
  
  # split into training and test dataset
  train = answers[1:num_train,]
  test = answers[(num_train+1):(h2o.nrow(answers)-1),]
  
  # columns for baseline ML
  predictors = c("Age",
                 "Gender")
  response = "AvsN"
  
  # create a XGBoost model (optimized gradient boosting model - supervised learning)
  baseline_xg = h2o.xgboost(x = predictors,
                            y = response,
                            training_frame = train,
                            validation_frame = test)
  
  baseline_gb = h2o.gbm(x = predictors,
                        y = response,
                        training_frame = train,
                        validation_frame = test,
                        stopping_metric = "AUC",
                        stopping_tolerance = 0.001,
                        score_tree_interval = 10,
                        stopping_rounds = 5)
  data("stop_words")
  
  # tokenized_LF = Q1
  tokenized_LF = h2o.tokenize(answers$LookFor, " ") %>%
    h2o.tolower()
  tokenized_LF = tokenized_LF[!(tokenized_LF %in% stop_words$word),]
  
  # tokenized_FT - Q2
  tokenized_FT = h2o.tokenize(answers$FreeT, " ") %>%
    h2o.tolower()
  tokenized_FT = tokenized_FT[!(tokenized_FT %in% stop_words$word),]
  
  tokenized = h2o.rbind(tokenized_LF,tokenized_FT)
  
  # train a word vector on total tokanized words
  w2v = h2o.word2vec(training_frame = tokenized)
  
  # create numeric answer vectors from wordvector
  LF_vecs = h2o.transform(word2vec = w2v,
                          words = tokenized_LF,
                          aggregate_method = "AVERAGE")
  FT_vecs = h2o.transform(word2vec = w2v,
                          words = tokenized_FT,
                          aggregate_method = "AVERAGE")
  
  # add wordvector answer columns to original data
  answers_ext = h2o.cbind(answers, LF_vecs, FT_vecs)
  
  # create new test and training datasets on extended data
  train_ext = answers_ext[1:num_train,]
  test_ext = answers_ext[(num_train+1):(h2o.nrow(answers_ext)-1),]
  
  # now train a XGBoost on all columns
  extended_xg = h2o.xgboost(y = response,
                            training_frame = train_ext,
                            validation_frame = test_ext)
  # now train a GBM on all columns
  extended_gb = h2o.gbm(y = response,
                        training_frame = train_ext,
                        validation_frame = test_ext,
                        stopping_metric = "AUC",
                        stopping_tolerance = 0.001,
                        score_tree_interval = 10,
                        stopping_rounds = 5)
  
  # analyse wordds
  uw = h2o.asfactor(tokenized_FT) %>%
    h2o.unique() %>%
    h2o.ascharacter()
  we = h2o.transform(w2v,
                     uw,
                     aggregate_method = "NONE")
  we = h2o.cbind(uw, we)
  
  ### prepare worddata for Q1
  uw_LF = h2o.asfactor(tokenized_LF) %>%
    h2o.unique() %>%
    h2o.ascharacter()
  we_LF = h2o.transform(w2v,
                        uw_LF,
                        aggregate_method = "NONE")
  we_LF = h2o.cbind(uw_LF,
                    we_LF)
  
  # use the values from the histogram for the threshold here
  lowC33 = we[we['C33'] < -0.002,]
  lowC33Words = head(lowC33['C1'],3) #top words for Q1 and Nike
  highC33 = we[we['C33'] > 0.002,]
  highC33Words = head(highC33['C1'],3)  #top words for Q1 and Addidas
  
  # use the values from the histogram for the threshold here
  lowC33_LF = we_LF[we_LF['C33'] < -0.002,]
  lowC33Words_LF = head(lowC33_LF['C1'],3) #top words for Q1 and Nike
  highC33_LF = we_LF[we_LF['C33'] > 0.002,]
  highC33Words_LF = head(highC33_LF['C1'],3)  #top words for Q1 and Addidas
  
  # use the values from the histogram for the threshold here
  lowC68 = we[we['C68'] < -0.003,]
  lowC68Words = head(lowC68['C1'],3) #top words for Q1 and Nike
  highC68 = we[we['C68'] > 0.003,]
  highC68Words = head(highC68['C1'],3)  #top words for Q1 and Addidas
  
  # use the values from the histogram for the threshold here
  lowC68_LF = we_LF[we_LF['C68'] < -0.002,]
  lowC68Words_LF = head(lowC68_LF['C1'],3) #top words for Q1 and Nike
  highC68_LF = we_LF[we_LF['C68'] > 0.003,]
  highC68Words_LF = head(highC68_LF['C1'],3)  #top words for Q1 and Addidas
  
  # Get the most important words from the classifications
  ImportantWords = data.frame(Q1_XG_Ad = highC68Words_LF,
                              Q1_XG_Nike = lowC68Words_LF,
                              Q2_XG_Ad = highC68Words,
                              Q2_XG_Nike = lowC68Words,
                              Q1_GB_Ad = highC33Words_LF,
                              Q1_GB_Nike = lowC33Words_LF,
                              Q2_GB_Ad = highC33Words,
                              Q2_GB_Nike = lowC33Words)
  colnames(ImportantWords) = c("Q1_XG_Ad",
                               "Q1_XG_Nike",
                               "Q2_XG_Ad",
                               "Q2_XG_Nike",
                               "Q1_GB_Ad",
                               "Q1_GB_Nike",
                               "Q2_GB_Ad",
                               "Q2_GB_Nike")
  
  ############ predict ust based on age
  Nike_LH_BA = data.frame()
  
  for (age in 1:60) {
    old_answer = as.h2o(data.frame("bla",
                                   "bla",
                                   age,
                                   "M",
                                   "N"))
    colnames(old_answer) = c("LookFor",
                             "FreeT",
                             "Age",
                             "Gender",
                             "AvsN")
    p = h2o.predict(baseline_xg,
                    old_answer)
    Nike_LH_BA = rbind(Nike_LH_BA,
                       data.frame(age = age,
                                  NLH = as.vector(unlist(p$N))[1]))
  }
  Nike_LH_BA$ALH = 1-Nike_LH_BA$NLH
  
  
  #######
  # create plots
  #######
  
  ### QUESTION  1
  abc <- bc[,1]
  abc <- na.omit(abc)
  colnames(abc) <- c('text')
  
  token_abc <- abc %>% 
    unnest_tokens(word,
                  text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=T)
  
  ### QUESTION 2
  
  bbc <- bc[,2]
  bbc <- na.omit(bbc)
  colnames(bbc) <- c('text')
  
  token_bbc <- bbc %>% 
    unnest_tokens(word,
                  text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=T)
  
  
  ### QUESTION 3
  cbc <- bc[,3]
  cbc <- na.omit(cbc)
  colnames(cbc) <- c('text')
  
  token_cbc <- cbc %>% 
    unnest_tokens(word,
                  text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=T)
  
  ### QUESTION 4
  dbc <- bc[,4]
  dbc <- na.omit(dbc)
  colnames(dbc) <- c('text')
  
  token_dbc <- dbc %>% 
    unnest_tokens(word,
                  text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=T)
  
  ### QUESTION 5
  ebc <- bc[,5]
  ebc <- na.omit(ebc)
  colnames(ebc) <- c('text')
  
  token_ebc <- ebc %>% 
    unnest_tokens(word,text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=T)
   
  output$histogram_1 <- renderPlot({
    freq_hist_a <- abc %>%
      unnest_tokens(word,
                    text) %>%
      anti_join(stop_words) %>%
      count(word,
            sort=TRUE) %>%
      ggplot(aes(word,
                 n))+
      geom_col()+
      xlab(NULL)+
      coord_flip()
    freq_hist_a
  })
  
  output$histogram_2 <- renderPlot({
    freq_hist_b <- bbc %>%
      unnest_tokens(word,
                    text) %>%
      anti_join(stop_words) %>%
      count(word,
            sort=TRUE) %>%
      ggplot(aes(word,
                 n))+
      geom_col()+
      xlab(NULL)+
      coord_flip()
    freq_hist_b
  })
  
  
  output$histogram_3 <- renderPlot({
    freq_hist_c <- cbc %>%
      unnest_tokens(word,
                    text) %>%
      anti_join(stop_words) %>%
      count(word,
            sort=TRUE) %>%
      ggplot(aes(word,
                 n))+
      geom_col()+
      xlab(NULL)+
      coord_flip()
    print(freq_hist_c)
  })

  output$histogram_4 <- renderPlot({
  freq_hist_d <- dbc %>%
    unnest_tokens(word,
                  text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=TRUE) %>%
    ggplot(aes(word,
               n))+
    geom_col()+
    xlab(NULL)+
    coord_flip()
  freq_hist_d
  })

  output$histogram_5 <- renderPlot({
  freq_hist_e <- ebc %>%
    unnest_tokens(word,
                  text) %>%
    anti_join(stop_words) %>%
    count(word,
          sort=TRUE) %>%
    ggplot(aes(word,
               n))+
    geom_col()+
    xlab(NULL)+
    coord_flip()
  freq_hist_e
  })

  output$wordcloud_1 <- renderPlot({
    token_abc %>%
      with(wordcloud(word,
                     n,
                     max.words = 100))
  })

  output$wordcloud_2 <- renderPlot({
    token_bbc %>%
      with(wordcloud(word,
                     n,
                     max.words = 100))
  })

  output$wordcloud_3 <- renderPlot({
    token_cbc %>%
      with(wordcloud(word,
                     n,
                     max.words = 100))
  })

  output$wordcloud_4 <- renderPlot({
    token_dbc %>%
      with(wordcloud(word,
                     n,
                     max.words = 100))
  })

  output$wordcloud_5 <- renderPlot({
    token_ebc %>%
      with(wordcloud(word,
                     n,
                     max.words = 100))
  })
  
  ## Baseline CGBoost plot
  output$BaselineXG <- renderPlot({
    h2o.varimp_plot(baseline_xg)
  })
  
  ## Baseline GBM Plot
  output$BaselineGBM <- renderPlot({
    h2o.varimp_plot(baseline_gb)
  })

  ## Extended XGBoost Plot
  output$ExtendedXG <- renderPlot({
    h2o.varimp_plot(extended_xg)
  })

  ## Extended GBM plot
  output$ExtendedGBM <- renderPlot({
    h2o.varimp_plot(extended_gb)
  })
  
  ## Age plot
  output$Age <- renderPlot({
  
  Nike_LH_BA %>%
    ggplot(aes(age,
               NLH))+ 
    geom_line(size = 1.1,
              alpha = 0.8,
              show.legend = FALSE)
  })
  
  ## Table
  output$imptwds <- renderTable({
    ImportantWords
  })
})