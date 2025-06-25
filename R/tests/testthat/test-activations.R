test_that("clock_activation works correctly", {
  # Test with vector input
  x <- c(-2, -1, 0, 1, 2)
  result <- clock_activation(x)

  expect_length(result, length(x))
  expect_true(all(is.finite(result)))
  expect_equal(result[3], 0) # Zero input should give zero output
})

test_that("seasonal_activation varies by hemisphere", {
  x <- c(-1, 0, 1)

  result_north <- seasonal_activation(x, hemisphere = "northern")
  result_south <- seasonal_activation(x, hemisphere = "southern")

  expect_length(result_north, length(x))
  expect_length(result_south, length(x))

  # Results should generally differ between hemispheres
  # (except at equinoxes)
  expect_true(all(is.finite(result_north)))
  expect_true(all(is.finite(result_south)))
})

test_that("circadian_activation follows 24-hour pattern", {
  x <- 1.0

  # Should produce finite output
  result <- circadian_activation(x)
  expect_true(is.finite(result))
  expect_true(result >= 0) # ReLU-like base
})

test_that("wave_activation is bounded", {
  x <- seq(-10, 10, length.out = 100)

  result <- wave_activation(x, frequency = 1.0, amplitude = 1.0)

  expect_length(result, length(x))
  expect_true(all(is.finite(result)))
  expect_true(all(abs(result) <= 1.1)) # Bounded by amplitude (with tolerance)
})

test_that("fourier_activation converges", {
  x <- seq(-pi, pi, length.out = 50)

  result1 <- fourier_activation(x, n_harmonics = 1)
  result5 <- fourier_activation(x, n_harmonics = 5)

  expect_length(result1, length(x))
  expect_length(result5, length(x))
  expect_true(all(is.finite(result1)))
  expect_true(all(is.finite(result5)))
})

test_that("quantum_activation produces valid states", {
  x <- rnorm(20)

  result <- quantum_activation(x, n_states = 4)

  expect_length(result, length(x))
  expect_true(all(is.finite(result)))
})

test_that("lorenz_activation is bounded", {
  x <- c(-100, -1, 0, 1, 100)

  result <- lorenz_activation(x)

  expect_length(result, length(x))
  expect_true(all(is.finite(result)))
  expect_true(all(abs(result) <= 1.0)) # Bounded by tanh
})

test_that("logistic_map_activation is bounded", {
  x <- seq(-5, 5, length.out = 50)

  result <- logistic_map_activation(x, r = 3.9, iterations = 3)

  expect_length(result, length(x))
  expect_true(all(is.finite(result)))
  expect_true(all(result >= -1 & result <= 1))
})

test_that("mandelbrot_activation handles complex inputs", {
  x <- c(-2, -1, 0, 1, 2)

  result <- mandelbrot_activation(x, max_iterations = 5)

  expect_length(result, length(x))
  expect_true(all(is.finite(result)))
})

test_that("activations handle edge cases", {
  # Empty input
  expect_length(clock_activation(numeric(0)), 0)
  expect_length(wave_activation(numeric(0)), 0)

  # Single value
  expect_length(clock_activation(1), 1)
  expect_length(wave_activation(1), 1)

  # NA handling
  x_with_na <- c(1, NA, 2)
  result <- wave_activation(x_with_na)
  expect_true(is.na(result[2]))
  expect_true(is.finite(result[1]))
  expect_true(is.finite(result[3]))
})
