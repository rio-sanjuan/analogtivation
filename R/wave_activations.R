#' Wave-Based Activation Functions
#'
#' @description
#' A collection of activation functions based on wave phenomena.

#' Wave Activation Function
#'
#' Composite waveform activation using sine and cosine components.
#'
#' @param x numeric vector of input values
#' @param frequency numeric; wave frequency (default: 1.0)
#' @param amplitude numeric; wave amplitude (default: 1.0)
#'
#' @return numeric vector of wave-activated values
#' @export
#'
#' @examples
#' wave_activation(c(-2, -1, 0, 1, 2))
#' wave_activation(c(-2, -1, 0, 1, 2), frequency = 2.0)
wave_activation <- function(x, frequency = 1.0, amplitude = 1.0) {
  # Composite of sine and cosine waves
  sine_component <- sin(frequency * x)
  cosine_component <- cos(frequency * x)
  
  # Weighted combination
  amplitude * (0.7 * sine_component + 0.3 * cosine_component)
}


#' Fourier Activation Function
#'
#' Fourier series-based activation function.
#'
#' @param x numeric vector of input values
#' @param n_harmonics integer; number of harmonics (default: 3)
#' @param weights numeric vector; harmonic weights (optional)
#'
#' @return numeric vector of Fourier-activated values
#' @export
#'
#' @examples
#' fourier_activation(c(-2, -1, 0, 1, 2))
#' fourier_activation(c(-2, -1, 0, 1, 2), n_harmonics = 5)
fourier_activation <- function(x, n_harmonics = 3, weights = NULL) {
  if (is.null(weights)) {
    weights <- rep(1, n_harmonics)
  }
  
  result <- rep(0, length(x))
  
  for (i in 1:n_harmonics) {
    harmonic <- i * x
    weight <- weights[i]
    result <- result + weight * sin(harmonic) / i
  }
  
  result
}


#' Quantum Activation Function
#'
#' Quantum-inspired activation with superposition of states.
#'
#' @param x numeric vector of input values
#' @param n_states integer; number of quantum states (default: 2)
#' @param state_amplitudes numeric vector; quantum state amplitudes (optional)
#'
#' @return numeric vector of quantum-activated values
#' @export
#'
#' @examples
#' quantum_activation(c(-2, -1, 0, 1, 2))
#' quantum_activation(c(-2, -1, 0, 1, 2), n_states = 3)
quantum_activation <- function(x, n_states = 2, state_amplitudes = NULL) {
  if (is.null(state_amplitudes)) {
    state_amplitudes <- rnorm(n_states, sd = 0.1)
  }
  
  # Normalize amplitudes (quantum normalization)
  normalized_amplitudes <- exp(state_amplitudes) / sum(exp(state_amplitudes))
  
  # Superposition of states
  result <- rep(0, length(x))
  
  for (i in 1:n_states) {
    phase <- 2 * pi * (i - 1) / n_states
    state_contribution <- normalized_amplitudes[i] * sin(x + phase)
    result <- result + state_contribution
  }
  
  result
}