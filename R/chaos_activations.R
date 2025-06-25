#' Chaos-Based Activation Functions
#'
#' @description
#' A collection of activation functions based on chaotic systems.

#' Lorenz Activation Function
#'
#' Lorenz attractor-inspired activation function.
#'
#' @param x numeric vector of input values
#' @param sigma numeric; Lorenz parameter sigma (default: 10.0)
#' @param rho numeric; Lorenz parameter rho (default: 28.0)
#' @param beta numeric; Lorenz parameter beta (default: 8/3)
#'
#' @return numeric vector of Lorenz-activated values
#' @export
#'
#' @examples
#' lorenz_activation(c(-2, -1, 0, 1, 2))
lorenz_activation <- function(x, sigma = 10.0, rho = 28.0, beta = 8/3) {
  # Simplified Lorenz dynamics applied to activation
  y <- tanh(x)  # Bounded transformation
  z <- 1 / (1 + exp(-x))  # Sigmoid transformation
  
  # One step of Lorenz dynamics
  dx <- sigma * (y - x)
  dy <- x * (rho - z) - y
  dz <- x * y - beta * z
  
  # Combine derivatives as activation
  0.1 * (dx + dy + dz)
}


#' Logistic Map Activation Function
#'
#' Logistic map chaos activation function.
#'
#' @param x numeric vector of input values
#' @param r numeric; logistic map parameter (default: 3.9)
#' @param iterations integer; number of iterations (default: 3)
#'
#' @return numeric vector of chaos-activated values
#' @export
#'
#' @examples
#' logistic_map_activation(c(-2, -1, 0, 1, 2))
logistic_map_activation <- function(x, r = 3.9, iterations = 3) {
  # Normalize inputs to [0, 1] range
  x_norm <- 1 / (1 + exp(-x))
  
  # Apply logistic map iterations
  for (i in 1:iterations) {
    x_norm <- r * x_norm * (1 - x_norm)
  }
  
  # Scale back to reasonable range
  2 * x_norm - 1
}


#' Mandelbrot Activation Function
#'
#' Mandelbrot set-inspired activation function.
#'
#' @param x numeric vector of input values
#' @param max_iterations integer; maximum iterations (default: 5)
#'
#' @return numeric vector of Mandelbrot-activated values
#' @export
#'
#' @examples
#' mandelbrot_activation(c(-2, -1, 0, 1, 2))
mandelbrot_activation <- function(x, max_iterations = 5) {
  # Use inputs as complex numbers (real part only)
  z <- complex(real = x, imaginary = 0)
  c <- complex(real = x * 0.5, imaginary = 0)  # Scale down c
  
  # Mandelbrot iteration
  for (i in 1:max_iterations) {
    z <- z * z + c
  }
  
  # Return magnitude, bounded
  tanh(Mod(z))
}