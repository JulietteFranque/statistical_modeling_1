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



find_sensitivity <- function(uplift, n_points_sim){
n_significant = 0
for (i in 1:n_points_sim) {
  df <- make_data(dates, uplift, covariate_1, covariate_2)
  w_uplift = df$Y_uplift
  data <- zoo(cbind(y=df$Y_uplift, covariate_1, covariate_2), dates)
  impact <- CausalImpact(data, pre_period, post_period)
  abs_lower = impact$summary$AbsEffect.lower[1]
  if(abs_lower > 0){
    n_significant = n_significant + 1
  }
}
return(n_significant / n_points_sim)
}

uplifts = c(0, 0.001, .01, 0.1, 0.25, 0.5, 1)

sensitivities = vector(length=length(uplifts))
for (i in 1:length(uplifts)) {
  print(i)
  sensitivities[i] = find_sensitivity(uplifts[i], 2^8)
}
df_sensitivities <- data.frame(uplifts, sensitivities)
df_sensitivities$se <- 2 * sqrt(df_sensitivities$sensitivities * (1 - df_sensitivities$sensitivities)/2^8)



ggplot(df_sensitivities) + 
  geom_point(aes(x=as.factor(uplifts*100), y=sensitivities)) +
geom_errorbar(aes(ymin = sensitivities - se, ymax = sensitivities + se, x=as.factor(uplifts* 100)), width=.2) + 
  xlab('effect size (%)') + ylab('proportion of intervals excluding zero')



n_points_sim <- 50
uplift <- 0.01
compaigns_lengths <- seq(from = 25, to = 150, by = 25)
ratios_CI = vector(length=length(compaigns_lengths))

for (i in 1:length(compaigns_lengths)) {
  number_within = 0
  for (k in 1:n_points_sim) { 
    dates <- seq(start_date, end_date + compaigns_lengths[i],"days")
    nums = seq(from=1, to=length(dates))
    covariate_1 <-   sin(2*pi*nums/90)
    covariate_2 <- sin(2*pi*nums/360)
    df <- make_data(dates, uplift=uplift, covariate_1=covariate_1, covariate_2=covariate_2, date_uplift=end_date)
    w_uplift = df$Y_uplift
    data <- zoo(cbind(y=df$Y_uplift, covariate_1, covariate_2), dates)
    pre_period = c(start_date, end_date)
    post_period = c(end_date + 1, end_date + compaigns_lengths[i])
    impact <- CausalImpact(data, pre_period, post_period, , model.args = list(dynamic.regression=TRUE))
    #print(plot(impact))
    lower_CI =  impact$summary['Average', 'RelEffect.lower']
    upper_CI = impact$summary['Average', 'RelEffect.upper']
    #print(lower_CI)
   # print(upper_CI)
   # print('--')
    if (uplift > lower_CI & uplift < upper_CI) {
      number_within = number_within + 1
    }
    ratios_CI[i] <- number_within / n_points_sim
    print(ratios_CI[i])
  }
}

df_CI <- data.frame(compaigns_lengths, ratios_CI)
df_CI$se <- 2 * sqrt(df_CI$ratios_CI * (1 - df_CI$ratios_CI)/n_points_sim)

ggplot(df_CI) + 
  geom_point(aes(x=as.factor(compaigns_lengths), y=ratios_CI)) +
  geom_errorbar(aes(ymin = ratios_CI - se, ymax = ratios_CI + se, x=as.factor(compaigns_lengths)), width=.2) + 
  xlab('effect size (%)') + ylab('proportion of intervals excluding zero')



