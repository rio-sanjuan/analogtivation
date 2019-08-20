###############################################
##
##  Analogtivation ~ Analyze Results
## 
##  author: Ryan Johnson
##  created: Aug 2019
##  
###############################################

## =========================================
## loading packages and environment settings
## =========================================

# Libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(hms)
  library(scales)
  library(glue)
  library(grid)
  library(jsonlite)
})

## =========================================
## load results & clean
## =========================================

# temp <- list.files("./logs", pattern = "*.json", full.names = TRUE)
# data <- dir("./logs", pattern = "*.json") %>% 
#   map_df(~jsonlite::fromJSON(paste0("[",paste0(readLines(file.path("./logs", .)), collapse=","),"]")))

data <- data.frame(
  val_loss = numeric(),
  loss = numeric(),
  epoch = numeric(),
  time = hms(),
  group = factor()
)

for (group in 1:24) {
  temp.data <- jsonlite::fromJSON(paste0("[",paste0(readLines(file.path("./logs", glue("log_{group}.json"))), collapse=","),"]"))
  
  data <- data %>% 
    tibble::add_row(
      val_loss = temp.data$val_loss,
      loss = temp.data$loss,
      epoch = temp.data$epoch,
      time = temp.data$time %>% hms::as_hms(),
      group = as.factor(group)
    )
  
  rm(temp.data)
}

## =========================================
## Draw a clock
## =========================================

# credit: https://www.stat.auckland.ac.nz/~paul/RG2e/interactive-clock.R
drawClock <- function(hour, minute) {
  t <- seq(0, 2*pi, length=13)[-13]
  x <- cos(t)
  y <- sin(t)
  
  # grid.newpage()
  pushViewport(dataViewport(x, y, gp=gpar(lwd=4)))
  
  # Circle with ticks
  grid.circle(x=0, y=0, default="native", 
              r=grid::unit(1, "native"))
  grid.segments(x, y, x*.9, y*.9, default="native")
  
  # Axis marks
  grid.segments(-1, 0, 1, 0,
                default="native", gp=gpar(lex=0.25, col = 'black'))
  grid.segments(0, -1, 0, 1,
                default="native", gp=gpar(lex=0.25, col = 'black'))
  
  # Minute hand
  minuteAngle <- pi/2 - (minute)/60*2*pi
  grid.segments(0, 0,
                .8*cos(minuteAngle + pi), .8*sin(minuteAngle + pi),
                default="native", gp=gpar(lex=1, col = 'gray'))
  grid.segments(0, 0, 
                .8*cos(minuteAngle), .8*sin(minuteAngle), 
                default="native", gp=gpar(lex=1))   
  
  # Hour hand
  hourAngle <- pi/2 - (hour + minute/60)/12*2*pi
  grid.segments(0, 0,
                .6*cos(hourAngle + pi), .6*sin(hourAngle + pi),
                default="native", gp=gpar(lex=2, col = 'gray'))
  grid.segments(0, 0, 
                .6*cos(hourAngle), .6*sin(hourAngle), 
                default="native", gp=gpar(lex=2))
  
  # Clock
  grid.circle(0, 0, default="native", r=unit(1, "mm"),
              gp=gpar(fill="white"))
}
drawClock(2, 35)

## =========================================
## plot results
## =========================================

# minute hand for positive x
# hour hand for negative x
for (gr in 1:24) {
  
  current.time <- data %>% 
    dplyr::filter(group == gr) %>% 
    dplyr::group_by(group) %>% 
    dplyr::summarize(Hour = mean(hour(time)) %% 12, Minute = mean(minute(time)))
  
  png(file = glue("images/{current.time$Hour}h{current.time$Minute}m.png"))
  
  # epoch performance
  g <- data %>% 
    dplyr::filter(group == gr) %>% 
    ggplot(aes(x = epoch, y = loss, group = group)) + geom_line() + theme_bw() + 
    labs(title = glue("Time of Run: {current.time$Hour}:{current.time$Minute}"))
  print(g)
  
  # add clock annotation
  pushViewport(viewport(x = 0.65, y = 0.6,
                        width = 0.3, height = 0.3,
                        just = c("left", "bottom")))
  grid.draw(rectGrob())
  grid.draw(drawClock(current.time$Hour, current.time$Minute))
  popViewport()
  
  dev.off()
}
