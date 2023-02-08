library(coolcat)


analys_personal_data_model_filename<-'./analys_personal_data_model.csv'
analys_personal_data_model <- read.csv(file=analys_personal_data_model_filename, 
                                       header = TRUE, stringsAsFactors = TRUE)
analys_personal_data_model = subset(analys_personal_data_model, select = -abandona)


c_clust<-coolcat(analys_personal_data_model,k=2,trace.log=T)
table(c_clust$clustering)
analys_personal_data_model$labels = c_clust$clustering
write.csv(analys_personal_data_model,'./analys_personal_data_clust.csv')


