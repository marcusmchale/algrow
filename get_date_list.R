
library(dplyr)
filelist<-data.frame(list=list.files(pattern = "\\.jpg$"))
filelist$list<-as.character(filelist$list)
filelist$date_hour<-(substr(filelist$list,(nchar(filelist$list)-24),(nchar(filelist$list)-11)))
filelist$date_hour_0<-paste(filelist$date_hour,"0",sep="")
filelist$date_hour_1<-paste(filelist$date_hour,"1",sep="")
filelist$date_hour_2<-paste(filelist$date_hour,"2",sep="")
filelist$date_hour_3<-paste(filelist$date_hour,"3",sep="")
filelist$date_hour_4<-paste(filelist$date_hour,"4",sep="")
filelist$date_hour_5<-paste(filelist$date_hour,"5",sep="")


filelist_f<-noquote(unique(c(filelist$date_hour_0,filelist$date_hour_1,filelist$date_hour_2,filelist$date_hour_3,filelist$date_hour_4,filelist$date_hour_5)))



write.table(filelist_f,"date_list.txt",row.names = F,quote=F,col.names = F)
