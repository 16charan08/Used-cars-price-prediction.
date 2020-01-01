library(tidyverse)
library(VIM)
library(forcats)
library(randomForest)
require(caTools)
library(lubridate)
library(glmnet)
library(earth)
library(mltools)
library(caret)
library(pls)
library(Metrics)
library(ROCR)

getmode <- function(v) {
  v <- v[!is.na(v)]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}




#cars <- subset(craigslistVehicles, price>1000)
carSample <- read.csv(file = "carSample1.csv")

#set.seed(1234)
#carSample <- sample_n(cars, size = 70000)
#cars <- as.data.frame(cars)
#craigslistVehicles <- NULL
#write_csv(carSample1,"carSample1.csv")

names(carSample)

carSample <- as.data.frame(carSample)

#Dimentionnality reduction
carSample <- subset(carSample,select = -c(city_url, VIN, image_url, desc))

#Removing features that have missingness more than 50%
missingness <- carSample %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()
carSample <- carSample %>% select(c(names(missingness[,missingness < 0.50])))


#Handling Outliers.
carSample <- subset(carSample, price < 100000)


train <- carSample %>% 
  mutate_if(is.character, as.factor) %>%
  mutate(id = as.numeric(url)) %>%
  mutate(city = fct_lump(city, n = 100)) %>%
  mutate(year = replace_na(year, getmode(year))) %>%
  mutate(year = fct_lump(as.factor(year), n = 30)) %>%
  mutate(manufacturer = replace_na(manufacturer, getmode(manufacturer))) %>%
  mutate(make = fct_lump(fct_explicit_na(make), n = 100)) %>%
  mutate(condition = fct_lump(fct_explicit_na(condition, na_level = "Other"), n=4)) %>%
  mutate(cylinders = fct_lump(fct_explicit_na(cylinders, na_level = "other"))) %>%
  mutate(fuel = replace_na(fuel, getmode(fuel))) %>%
  mutate(odometer = replace_na(odometer, mean(odometer, na.rm = TRUE))) %>%
  mutate(log_odometer = log(odometer+1)) %>%
  mutate(title_status = replace_na(title_status, getmode(title_status))) %>%
  mutate(transmission = replace_na(transmission, getmode(transmission))) %>%
  mutate(drive = fct_explicit_na(drive,na_level = "Other"))%>%
  mutate(type = fct_explicit_na(type, na_level = "other")) %>%
  mutate(paint_color = fct_explicit_na(paint_color, na_level = "other")) %>%
  mutate(log_price = log(price+1)) %>%
  mutate(lat= replace_na(lat, mean(lat, na.rm = TRUE))) %>%
  mutate(long = replace_na(long, mean(long, na.rm=TRUE))) %>%
  subset(select = -c(odometer,price, url, city)) 



#summary(train$year)
train %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()  

model <- lm(log_price ~ ., data = train)

set.seed(123) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model <- train(log_price ~ ., data = train, method = "lm",
               trControl = train.control)

pred <- predict(model, train)
rmse(pred, train$log_price)
par(mfrow = c(2,2))
plot(model)

#using MARS model
?earth

#Train the model
marsFit <- earth(log_price ~ ., 
                 data = train,
                 degree=2,nk=48,pmethod="cv",nfold=10,ncross=5)

mpred <- predict(marsFit, train)
rmse(train$log_price,marsFit$fitted.values)
head(marsFit$fitted.values)
head(train$log_price)

#custom control parameters




#Ridge Regression

set.seed(1234)
ridge <- train(log_price ~ ., 
               data = train, method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0, lambda = seq(-2,2,length=5)), trControl = custom)
rpred <- predict(ridge, train)
rmse(train$log_price,rpred)
summary(ridge)
plot(ridge)

#Lasso
custom <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = T)
set.seed(1234)
#Train the model
lasso <- train(log_price ~ ., 
               data = train, method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001,1,length=5)), trControl = custom)

lpred <- predict(lasso, train)
rmse(lpred, train$log_price)
plot(lasso)