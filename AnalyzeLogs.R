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
  library(jsonlite)
})

## =========================================
## load results & clean
## =========================================

temp <- list.files("./logs", pattern = "*.json", full.names = TRUE)

data <- dir("./logs", pattern = "*.json") %>% 
  map_df(~jsonlite::fromJSON(paste0("[",paste0(readLines(file.path("./logs", .)), collapse=","),"]")))

data <- data.frame(
  val_loss = numeric(),
  loss = numeric(),
  epoch = numeric(),
  time = character(),
  group = numeric()
)

for (group in 1:24) {
  temp.data <- jsonlite::fromJSON(paste0("[",paste0(readLines(file.path("./logs", glue("log_{group}.json"))), collapse=","),"]"))
  
  data <- data %>% 
    tibble::add_row(
      val_loss = temp.data$val_loss,
      loss = temp.data$loss,
      epoch = temp.data$epoch,
      time = temp.data$time,
      group = group
    )
  
  rm(temp.data)
}


## =========================================
## plot results
## =========================================

data %>% 
  dplyr::group_by(group) %>% 
  dplyr::summarize()


data %>% 
  ggplot(aes(x = epoch, color = as.factor(group))) + 
  geom_line(aes(y = loss))


## =========================================
## plot results
## =========================================

# Generate digitial clock face
first.nine <- c('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12')
hours <- c(first.nine, as.character(seq(10,23,1)))
mins <- c(first.nine, as.character(seq(10,59,1)))
time.chars.l <- lapply(hours, function(h) paste(h, ':', mins, sep=''))
time.chars <- do.call(c, time.chars.l)

# Generate analog clock face
hour.pos <- seq(0, 12, 12/(12*60))[1:720]
min.pos <-seq(0,12, 12/60)[1:60]
all.hours <- rep(hour.pos, 2)
all.times <- cbind(all.hours, min.pos, 24)
for(i in 1:nrow(all.times)) {
  
  png(glue("images/clocks/{time.chars[i]}:00.jpeg"))
  
  cur.time <- data.frame(list(times=c(all.times[i,1], all.times[i,2]), hands=c(.5, 1)))
  clock <- ggplot(cur.time, aes(xmin=times, xmax=times+0.03, ymin=0, ymax=hands))+
    geom_rect(aes(alpha=0.8))+
    scale_x_continuous(limits=c(0,all.hours[length(all.hours)]), breaks=0:11, 
                       labels=c(12, 1:11))+
    scale_y_continuous(limits=c(0,1.1))+scale_alpha()+theme_bw()+
    coord_polar() + labs(title = time.chars[i]) + 
    theme(axis.text.y=element_blank(), axis.ticks=element_blank(),
          panel.grid.major=element_blank(),
          strip.background = element_rect(colour = 'white'),
          legend.position = "none")
  # Save images to a folder names 'clocks'
  dev.off()

  
  # ggsave(plot=clock, path = "./images/clocks", filename=paste0(time.chars[i],":00.jpeg"), 
  #        height=5, width=5, device = "jpeg")
}


current_time = "20:20:22" %>% hms::as_hms()

hour = current_time %>% hour()
minute = current_time %>% minute()
second = current_time %>% second()







