# Take ImageJ output and convert to mm2
'%>%' <- magrittr::'%>%'

imagej_out <- file.path('sample_data/size.txt') # tsv from ImageJ
sample_details <- file.path("sample_data/sample_list.txt")
discs_per_tank <- 48
size_filename <- file.path('sample_output/size.txt')

size_df <-read.table(
  imagej_out,
  sep="\t",
  header=F,
  stringsAsFactors = F,
  colClasses = c("character","character","numeric","numeric","integer"),
  col.names = c("no","pic","area","perc_area","slice")
) %>%
  dplyr::mutate(tank = as.factor(substr(pic,nchar(pic),nchar(pic))))  # note this won't work for more than 9 tanks

#extract stack depth (number of pictures for each tank beacause can be different and important when we will merge
#the setup dataframe+ imageJ dataframe )
depths <- c("1","2","3") %>%
  tibble::as_tibble_col(column_name='tank') %>%
  dplyr::rowwise() %>%
  dplyr::mutate(depth = nrow(size_df[size_df$tank==tank,])/discs_per_tank)

wells_df <- data.frame(
  tank=rep(unique(size_df$tank), each=discs_per_tank),
  disc= rep(1:discs_per_tank, times=length(unique(size_df$tank)))
) %>%
  dplyr::mutate(well = paste(disc,tank, sep=".")) %>%
  dplyr::left_join(depths, by="tank") %>%
  dplyr::rowwise() %>%
  dplyr::slice(rep(1:dplyr::n(), each=depth)) %>%
  dplyr::select(-tank)

size_df <- cbind(wells_df, size_df) %>%
  dplyr::mutate(
    size = (100-perc_area)*area/100,
    time = as.POSIXct(strptime(substr(pic,(nchar(pic)-20),(nchar(pic)-5)),format="%Y-%m-%d_%Hh%M"))
  ) %>%
  dplyr::select(c(well, size, time))

#load sample details
gtp_list<-read.table(
  sample_details,
  sep="\t",
  header=T,
  colClasses = c(rep("character",8))
)

size_df <- size_df %>%
  dplyr::left_join(gtp_list, by='well')

write.table(size_df, size_filename, sep='\t', row.names = F)



