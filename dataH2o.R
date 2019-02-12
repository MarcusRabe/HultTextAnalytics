######################################################################################
# Playing around with some machine learning on the data captured during the
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
# Author: Marcus Rabe
######################################################################################

library(h2o)

# initialise H2O library
h2o.init()

# build path to data file
path = paste(dirname(rstudioapi::getSourceEditorContext()$path), "/Data2.csv", sep = "")

# import data file
answers = h2o.importFile(path, header = TRUE, col.names = c("LookFor","FreeT","Age","Gender","AvsN"))


num_train = 45 # records in training dataset

# split into training and test dataset
train = answers[1:num_train,]
test = answers[(num_train+1):(h2o.nrow(answers)-1),]

# columns for baseline ML
predictors = c("Age", "Gender")
response = "AvsN"

# create a XGBoost model (optimized gradient boosting model - supervised learning)
baseline_xg = h2o.xgboost(x = predictors,
                          y = response,
                          training_frame = train,
                          validation_frame = test)
h2o.confusionMatrix(baseline_xg)
h2o.varimp_plot(baseline_xg)

# create a Gradient Boosting model (supervised learning)
baseline_gb = h2o.gbm(x = predictors,
                      y = response,
                      training_frame = train,
                      validation_frame = test,
                      stopping_metric = "AUC",
                      stopping_tolerance = 0.001,
                      score_tree_interval = 10,
                      stopping_rounds = 5)
h2o.confusionMatrix(baseline_gb)
h2o.varimp_plot(baseline_gb)

# tokenize word columns
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

# test word vektor
# h2o.findSynonyms(w2v, "style", 5)

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
h2o.confusionMatrix(extended_xg)
h2o.varimp_plot(extended_xg)

# GBM on all columns
extended_gb = h2o.gbm(y = response,
                      training_frame = train_ext,
                      validation_frame = test_ext,
                      stopping_metric = "AUC",
                      stopping_tolerance = 0.001,
                      score_tree_interval = 10,
                      stopping_rounds = 5)
h2o.confusionMatrix(extended_gb)
h2o.varimp_plot(extended_gb)

### prepare worddata for Q2
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

### look for highest predictors for Q2 GB (C33 is from VarImpPlot)
h2o.hist( we['C33'])

# use the values from the histogram for the threshold here
lowC33 = we[we['C33'] < -0.002,]
lowC33Words = head(lowC33['C1'],3) #top words for Q1 and Nike
highC33 = we[we['C33'] > 0.002,]
highC33Words = head(highC33['C1'],3)  #top words for Q1 and Addidas

### look for highest predictors for Q1 GB (C33 is from VarImpPlot)
h2o.hist( we_LF['C33'])

# use the values from the histogram for the threshold here
lowC33_LF = we_LF[we_LF['C33'] < -0.002,]
lowC33Words_LF = head(lowC33_LF['C1'],3) #top words for Q1 and Nike
highC33_LF = we_LF[we_LF['C33'] > 0.002,]
highC33Words_LF = head(highC33_LF['C1'],3)  #top words for Q1 and Addidas

#########
#XGBoost
h2o.varimp_plot(extended_xg)

### look for highest predictors for Q2 XG (C68 is from VarImpPlot)
h2o.hist( we['C68'])

# use the values from the histogram for the threshold here
lowC68 = we[we['C68'] < -0.003,]
lowC68Words = head(lowC68['C1'],3) #top words for Q1 and Nike
highC68 = we[we['C68'] > 0.003,]
highC68Words = head(highC68['C1'],3)  #top words for Q1 and Addidas

### look for highest predictors for Q1 GB (C68 is from VarImpPlot)
h2o.hist( we_LF['C68'])

# use the values from the histogram for the threshold here
lowC68_LF = we_LF[we_LF['C68'] < -0.002,]
lowC68Words_LF = head(lowC68_LF['C1'],3) #top words for Q1 and Nike
highC68_LF = we_LF[we_LF['C68'] > 0.003,]
highC68Words_LF = head(highC68_LF['C1'],3)  #top words for Q1 and Addidas

############
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

# plot the likelihood of someone buying Nike by age
Nike_LH_BA %>%
  ggplot(aes(age,
             NLH))+ 
  geom_line(size = 1.1,
            alpha = 0.8,
            show.legend = FALSE)

#################