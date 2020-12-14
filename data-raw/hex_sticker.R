p <-
  create_board(.seed = 42) %>%
  print_board() +
  ggplot2::annotate("rect", xmin = 3.53, xmax = 6.47, ymin = 5.53, ymax = 6.47, alpha = 1, fill = "#eeeeee") +
  ggplot2::annotate("text", x = 5, y = 6, size = 8, label = "pseudo-ku")

ggplot2::ggsave("man/figures/base.png")
hexSticker::sticker(
  "man/figures/base.png"
  , package=""
  , h_fill="#FFF"
  , h_color="#424242"
  , s_x=1
  , s_y=1
  , s_width=0.9
  , s_height=0.6
  , filename="man/figures/analogtivation.png"
  , white_around_sticker = TRUE
)
