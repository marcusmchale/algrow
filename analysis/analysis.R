# Take size and calculate lsgr
'%>%' <- magrittr::'%>%'

size_filepath <- file.path("sample_output/size.txt")
hour_for_daily_rgr <- 14
sgr_filepath <- file.path('sample_output/daily_sgr.txt')

size_df <- read.table(
  size_filepath,
  sep="\t",
  header=T,
  stringsAsFactors = F,
  colClasses = c("character","numeric", rep("character", 8))
) %>%
  dplyr::mutate(time = as.POSIXct(time))

daily_size <- size_df %>%
  dplyr::mutate(
    hour = as.numeric(format(time, "%H")),
    date = format(time,format="%Y-%m-%d")
  ) %>%
  dplyr::filter(hour == hour_for_daily_rgr) %>%
  dplyr::select(-hour) %>%
  dplyr::group_by(gtp, date, tank, sample.ID, well) %>%
  dplyr::summarise_at("size", mean, na.omit=T) %>%
  dplyr::ungroup()

startdate <- min(daily_size$date)

daily_size <- daily_size %>%
  dplyr::group_by(well) %>%
  dplyr::mutate(
    prior.size = dplyr::lag(size, n=1, default=NA, order_by=date),
    lsgr = 100*(log(size)-log(prior.size))
  ) %>%
  dplyr::ungroup()

write.table(daily_size, sgr_filepath, row.names=F, sep="\t")

wells_to_remove <- c(
  "16.1","20.1","45.1","48.1",
  "14.2","20.2","21.2","24.2","38.2","39.2","41.2","42.2",
  "1.3", "4.3","9.3","12.3","26.3", "29.3","36.3"
)

daily_size_filtered <- daily_size %>%
  dplyr::mutate(day = difftime(date, startdate, units='days')) %>%
  dplyr::filter(
    !(well %in% wells_to_remove),
    day > 2,
    size > 0
  ) %>%
  dplyr::group_by(date, gtp, tank, sample.ID) %>%
  dplyr::summarise(lsgr = mean(lsgr)) %>%
  dplyr::ungroup()

ggplot2::ggplot(daily_size_filtered, ggplot2::aes(y=lsgr, colour=gtp)) +
  ggplot2::geom_boxplot() +
  ggplot2::facet_wrap(~date, ncol=1)

overall_sgr <- daily_size_filtered %>%
  dplyr::group_by(gtp, tank, sample.ID) %>%
  dplyr::summarise(lsgr = mean(lsgr)) %>%
  dplyr::ungroup()

p <- ggplot2::ggplot(overall_sgr, ggplot2::aes(x=gtp, y=lsgr, colour=gtp)) +
  ggplot2::geom_boxplot() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust=1))

ggplot2::ggsave()


dplyr::filter(overall_sgr, gtp=="RUS24")
dplyr::filter(overall_sgr, gtp=="RUS22")

anova(lm(lsgr~gtp, overall_sgr))


