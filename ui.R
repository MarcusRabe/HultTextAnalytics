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
# Author: Marcus Rabe and Team 12
######################################################################################

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("NIKE VS ADIDAS CONSUMER ANALYSIS"),
  
    # Show a plot of the generated distribution
    mainPanel(
      tabsetPanel(type = "tabs",
                 tabPanel("Q1: Quality",
                          plotOutput("histogram_1"),
                          plotOutput("wordcloud_1")),
                 tabPanel("Q2: Activity",
                          plotOutput("histogram_2"),
                          plotOutput("wordcloud_2")),
                 tabPanel("Q3: Age",
                          plotOutput("histogram_3"),
                          plotOutput("wordcloud_3")),
                 tabPanel("Q4: Gender",
                          plotOutput("histogram_4"),
                          plotOutput("wordcloud_4")),
                 tabPanel("Q5:Brand",
                          plotOutput("histogram_5"),
                          plotOutput("wordcloud_5")),
                 tabPanel("Baseline GBM",
                          plotOutput("BaselineGBM")),
                 tabPanel("Baseline XGBoost",
                          plotOutput("BaselineXG")),
                 tabPanel("Extended GBM",
                          plotOutput("ExtendedGBM")),
                 tabPanel("Extended XGBoost",
                          plotOutput("ExtendedXG")),
                 tabPanel("Age",
                          plotOutput("Age")),
                 tabPanel("Word Relations",
                          tableOutput("imptwds"))
      )
    )
  )
)
