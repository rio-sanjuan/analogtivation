#' Clock Activation Function
#'
#' Activation function that changes behavior based on current system time.
#' Uses minute hand angle for positive inputs and hour hand angle for negative inputs.
#'
#' @param x numeric vector of input values
#' 
#' @return numeric vector of activated values
#' @export
#' 
#' @examples
#' clock_activation(c(-2, -1, 0, 1, 2))
clock_activation <- function(x) {
  current_time <- Sys.time()
  
  hour <- as.numeric(format(current_time, "%I"))
  minute <- as.numeric(format(current_time, "%M"))
  second <- as.numeric(format(current_time, "%S"))
  
  # Calculate exact positions
  exact_hour <- hour + minute/60 + second/3600
  exact_minute <- minute + second/60
  
  # Convert to angles (clock notation)
  hour_angle <- -1 * (360 * exact_hour/12 - 90)
  minute_angle <- -1 * (360 * exact_minute/60 - 90)
  
  # Calculate slopes
  hour_slope <- tan(hour_angle * pi/180)
  minute_slope <- tan(minute_angle * pi/180)
  
  # Apply different slopes based on sign
  ifelse(x >= 0, minute_slope * x, hour_slope * x)
}


#' Seasonal Activation Function
#'
#' Activation function that varies with the seasons.
#'
#' @param x numeric vector of input values
#' @param hemisphere character; either "northern" or "southern"
#' 
#' @return numeric vector of seasonally adjusted activation
#' @export
#' 
#' @examples
#' seasonal_activation(c(-2, -1, 0, 1, 2))
seasonal_activation <- function(x, hemisphere = "northern") {
  day_of_year <- as.numeric(format(Sys.Date(), "%j"))
  
  # Adjust for hemisphere
  if (hemisphere == "southern") {
    day_of_year <- (day_of_year + 182) %% 365
  }
  
  # Calculate seasonal factor (peaks in summer, troughs in winter)
  seasonal_factor <- sin(2 * pi * (day_of_year - 80) / 365)
  
  # Apply seasonal modulation
  x * (1 + 0.3 * seasonal_factor)
}


#' Circadian Activation Function
#'
#' 24-hour circadian rhythm activation function.
#'
#' @param x numeric vector of input values
#' 
#' @return numeric vector of circadian-modulated activation
#' @export
#' 
#' @examples
#' circadian_activation(c(-2, -1, 0, 1, 2))
circadian_activation <- function(x) {
  current_time <- Sys.time()
  hour <- as.numeric(format(current_time, "%H"))
  minute <- as.numeric(format(current_time, "%M"))
  
  # Convert to decimal hours
  decimal_hour <- hour + minute/60
  
  # Circadian rhythm (peaks around 2pm, troughs around 3am)
  circadian_factor <- sin(2 * pi * (decimal_hour - 6) / 24)
  
  # Apply circadian modulation with ReLU-like base
  base_activation <- pmax(0, x)
  base_activation * (1 + 0.2 * circadian_factor)
}