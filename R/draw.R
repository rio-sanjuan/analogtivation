#' Draw analog clock visualization
#'
#' Creates a visualization of an analog clock showing the specified time
#' or current time if not specified.
#'
#' @param time POSIXct object or NULL for current time
#' @param show_activation logical; whether to show activation function overlay
#'
#' @return ggplot2 object
#' @export
#'
#' @examples
#' draw_clock()
#' draw_clock(as.POSIXct("2023-06-15 14:30:00"))
draw_clock <- function(time = NULL, show_activation = FALSE) {
  if (is.null(time)) {
    time <- Sys.time()
  }

  hour <- as.numeric(format(time, "%I"))
  minute <- as.numeric(format(time, "%M"))
  second <- as.numeric(format(time, "%S"))

  # Calculate angles (clock starts at 12)
  hour_angle <- (hour + minute / 60 + second / 3600) * 30 - 90
  minute_angle <- (minute + second / 60) * 6 - 90

  # Create clock face
  clock_df <- data.frame(
    x = cos(seq(0, 2 * pi, length.out = 12) - pi / 2),
    y = sin(seq(0, 2 * pi, length.out = 12) - pi / 2),
    label = 1:12
  )

  # Create hands
  hands_df <- data.frame(
    hand = c("hour", "minute"),
    x_end = c(
      0.5 * cos(hour_angle * pi / 180),
      0.8 * cos(minute_angle * pi / 180)
    ),
    y_end = c(
      0.5 * sin(hour_angle * pi / 180),
      0.8 * sin(minute_angle * pi / 180)
    ),
    size = c(3, 2)
  )

  p <- ggplot2::ggplot() +
    ggplot2::geom_point(
      data = clock_df,
      ggplot2::aes(x = x, y = y),
      size = 3
    ) +
    ggplot2::geom_text(
      data = clock_df,
      ggplot2::aes(x = x * 0.9, y = y * 0.9, label = label),
      size = 5
    ) +
    ggplot2::geom_segment(
      data = hands_df,
      ggplot2::aes(
        x = 0, y = 0, xend = x_end, yend = y_end,
        size = size, color = hand
      ),
      arrow = ggplot2::arrow(length = ggplot2::unit(0.3, "cm"))
    ) +
    ggplot2::scale_size_identity() +
    ggplot2::scale_color_manual(values = c("hour" = "blue", "minute" = "red")) +
    ggplot2::coord_fixed() +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "bottom",
      axis.text = ggplot2::element_blank(),
      axis.title = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank()
    ) +
    ggplot2::ggtitle(format(time, "%H:%M:%S"))

  if (show_activation) {
    # Add activation function visualization
    x_vals <- seq(-2, 2, length.out = 100)
    y_vals <- clock_activation(x_vals)
    activation_df <- data.frame(x = x_vals, y = y_vals)

    p <- p + ggplot2::geom_line(
      data = activation_df,
      ggplot2::aes(x = x, y = y),
      color = "green", alpha = 0.5, size = 1.5
    )
  }

  return(p)
}
