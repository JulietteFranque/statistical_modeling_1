library(CausalImpact)
library(ggplot2)
library(bsts)
library(dplyr)
library(reshape2)


start_date <- as.Date("2013/01/01")
end_date <- as.Date("2014/06/30")
date_uplift = as.Date("2014/01/01")
dates <- seq(start_date, end_date,"days")
pre_period = c(start_date, date_uplift)
post_period = c(date_uplift + 1, end_date)
n_points <- length(dates)
nums = seq(from=1, to=n_points)
covariate_1 <-   sin(2*pi*nums/90)
covariate_2 <- sin(2*pi*nums/360)

make_data <- function(dates, uplift, covariate_1, covariate_2, date_uplift=as.Date("2014/01/01")) {
  n_points <- length(dates)
  nums = seq(from=1, to=n_points)
  n_points = length(dates)
  dim <- 3
  T <- diag(dim)
  Q <- diag(dim) * 0.01 # mu, beta_1, beta_2
  alpha0 <- c(10,0,0)
  alpha <- matrix(ncol=dim, nrow=n_points)
  alpha[1,] <- alpha0%*%T + mvrnorm(n=1, mu=rep(0, dim), Sigma=Q)
  
  for (i in 2:n_points) {
    alpha[i,] <- alpha[(i-1),]%*%T + mvrnorm(n=1, mu=rep(0, dim), Sigma=Q)
  }
  
  Z <- matrix(nrow=dim, ncol=n_points)
  Z[1, ] <- rep(1, n_points)
  Z[2, ] <-  covariate_1
  Z[3, ] <- covariate_2
  
  Y = rowSums(t(Z) * alpha, dims=1) + sqrt(0.01)*rnorm(1)
  Y_uplift <- Y
  Y_uplift[dates > date_uplift] <- Y_uplift[dates > date_uplift] * (1 + uplift)
  df <- data.frame(dates, Y, Y_uplift)
  return(df)
}

df = make_data(dates, 0.2, covariate_1, covariate_2)
data <- zoo(cbind(y=df$Y_uplift, covariate_1, covariate_2), dates)
pre_period = c(start_date, end_date)
post_period = c(end_date + 1, end_date + compaigns_lengths[i])
impact <- CausalImpact(data, pre_period, post_period)



df$Y_na <- df$Y
df$Y_na[dates > date_uplift] <- NA
ss <- AddLocalLevel(list(), df$Y_na)

bsts.model <- bsts(df$Y_na ~ covariate_1, ss, niter = 1000)

