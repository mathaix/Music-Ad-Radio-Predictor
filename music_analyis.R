library(class)
library(ggplot2)
library(scatterplot3d)
library(boot)
data_path = "/Users/mathaix/dev/work/project/radio_data.csv"
data.cols= c("filename","beats", "mean","low", "high", "category" )
radio_data <- read.csv(data_path, sep=",", row.names=NULL, col.name = data.cols)
summary(radio_data)
ggplot(radio_data, aes(x=low/mean, y=beats)) + geom_point(aes(colour=category)) + ggtitle("Distribution low/mean vs beats")
ggplot(radio_data, aes(x=low/mean, y=high/low)) + geom_point(aes(colour=category)) + ggtitle("Distribution high/low vs low/mean")
attach(radio_data)

lrfit.1 <- glm( category ~ low/mean  +  low/high + high + low , family= binomial(link="logit"), data = radio_data)
summary(lrfit.1)

lrfit.2 <- glm( category ~ mean/low + mean + low , family= binomial(link="logit"), data = radio_data)
summary(lrfit.2)



#cross validation -1
cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)
(cv.err <- cv.glm(radio_data,  lrfit.1, cost, K = 8)$delta)

#cross validation -2
(cv.err <- cv.glm(radio_data,  lrfit.2, cost, K = 8)$delta)

#based on error rate. model 1 appears to be better


