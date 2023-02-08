if(!require('cba')) {
  install.packages('cba')
  library('cba')
}

analys_personal_data_model_filename<-'./analys_personal_data_model.csv'
analys_personal_data_model <- read.csv(file=analys_personal_data_model_filename, 
                                       header = TRUE, stringsAsFactors = TRUE)
x  = subset(analys_personal_data_model, select = -abandona)
x <- as.dummy(x)
rc <-rockCluster(x, n=2, theta=0.125)
print(rc)
rf <- fitted(rc)
table(analys_personal_data_model$abandona, rf$cl)

analys_personal_data_clust = as.data.frame(x)

analys_personal_data_clust$labels <- rf$cl
analys_personal_data_clust$abandona <- analys_personal_data_model$abandona
write.csv(analys_personal_data_clust,'analys_personal_data_clust.csv')


