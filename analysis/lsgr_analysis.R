# Take size and calculate lsgr
'%>%' <- magrittr::'%>%'

sample_details <- read.table(
  "sample_details.txt",
  sep="\t",
  stringsAsFactors = F,
  header=T,
  colClasses = rep("character")
) %>%
  dplyr::select(tank, number, strain) %>%
  dplyr::mutate(strain = factor(strain, levels = c("BLD17", "TANK1", paste0("LOG",1:24)))) %>%
  droplevels() %>%
  dplyr::rename(well = number)


size <- read.table(
  "area.csv",
  sep=",",
  stringsAsFactors = F,
  header=T,
  colClasses = c(rep("character", 3), rep("numeric",2))
) %>%
  dplyr::mutate(filename = substr(filename, 25, length(filename))) %>%
  tidyr::separate(filename, c("date", "time", "tank"), "_") %>%
  dplyr::mutate(
    time = as.POSIXct(paste(date, time), format="%Y-%m-%d %Hh%Mm"),
    date = as.Date(date, format="%Y-%m-%d"),
    hour = lubridate::hour(time),
    tank = substr(tank, 3,3)
  ) %>%
  dplyr::rename(size = mm.) %>%
  dplyr::left_join(sample_details) %>%
  dplyr::group_by(date, hour, tank, well, strain) %>%
  dplyr::summarise(size = median(as.numeric(size))) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(day = date-min(date))


daily_lsgr <- size %>%
  dplyr::filter(hour == 14) %>%
  dplyr::select(-hour) %>%
  dplyr::group_by(tank, well, strain) %>%
  dplyr::mutate(
    prior.size = dplyr::lag(size, n=1, default=NA, order_by=date),
    lsgr = 100*(log(size)-log(prior.size))
  ) %>%
  dplyr::ungroup() %>%
  dplyr::filter(
    day > 1
  )

p <- ggplot2::ggplot(daily_lsgr, ggplot2::aes(y=lsgr, x=strain, colour=strain)) +
  ggplot2::geom_boxplot() +
  ggplot2::ylim(-5,30) +
  ggplot2::facet_wrap(~date, ncol=1) +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust=1))
p
ggplot2::ggsave("daily_sgr.png", p)

sgr <- daily_lsgr %>%
  dplyr::filter(day > 2) %>%
  dplyr::group_by(tank, well, strain) %>%
  dplyr::summarise(sgr = mean(lsgr)) %>%
  dplyr::ungroup()

p2 <- ggplot2::ggplot(sgr, ggplot2::aes(x=strain, y=sgr, colour=strain)) +
  ggplot2::geom_boxplot() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust=1))
p2
ggplot2::ggsave("sgr.png", p2)
