library(mirt)


data <- read.csv("~/llmbenchmark/leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv")

data_clean <- na.omit(data)              # remove NA rows
data_clean <- data_clean[, colSums(is.na(data_clean)) == 0]  # remove NA columns

hist(data_clean[,2])
write.csv(data_clean, "~/llmbenchmark/cleaned_leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv", row.names = FALSE)


# Fit the model with improved convergence settings
str(data[1:50,-1])

data <- expand.table(LSAT7)
All <- cbind(data, data, data, data, data)
colnames(All)<-c("q1", "q2", "q3", "q4", "q5","q6", "q7", "q8", "q9", "q10","q11", "q12", "q13", "q14", "q15","q16", "q17", "q18", "q19", "q20","q21", "q22", "q23", "q24", "q25")

mod <- mirt(All, 1)

########################
model <- mirt(data_clean[,2:15], 1, itemtype = "3PL", 
              technical = list(NCYCLES = 3000),  # Allow more MHRM cycles if needed
              na.rm = TRUE
)

print(model)

cat("\nItem Parameters:\n")
item_params <- coef(model, simplify = TRUE)$items
print(item_params)

cat("\nPerson Theta:\n")
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE)

# # Save results to files
benchmark_name <- "3"
output_params_file <- paste0("irt_test_item_parameters_", benchmark_name, ".csv")
output_scores_file <- paste0("irt_test_person_scores_", benchmark_name, ".csv")

write.csv(item_params, output_params_file, row.names = TRUE)
write.csv(theta_scores, output_scores_file, row.names = TRUE)



#######################
df1 <- read.csv("irt_test_item_parameters_1.csv", stringsAsFactors = FALSE)
df2 <- read.csv("irt_test_item_parameters_2.csv", stringsAsFactors = FALSE)
df3 <- read.csv("irt_test_item_parameters_3.csv", stringsAsFactors = FALSE)

pf1 <- read.csv("irt_test_person_scores_1.csv", stringsAsFactors = FALSE)
pf2 <- read.csv("irt_test_person_scores_2.csv", stringsAsFactors = FALSE)
pf3 <- read.csv("irt_test_person_scores_3.csv", stringsAsFactors = FALSE)
# Check if identical (same values, same order, same column names)
identical(df1, df2)
identical(df1,df3)
identical(pf1,pf2)
identical(pf2,pf3)
M2(model)
itemfit(model)
library(cowplot)
library(ggmirt)
library(tidyverse)

itempersonMap(model)

tracePlot(model)
itemInfoPlot(model, facet = T)
# 全局拟合的折中：M2*，但降耗
M2(model, type="M2*", calcNull=FALSE, quadpts=21, QMC=FALSE)

# 看局部依赖（题目对的残差相关/LD 统计）
q3  <- residuals(model, type = "Q3")
ldg <- residuals(model, type = "LDG2", df.p=TRUE)

# item fit
it <- itemfit(model, fit_stats = "S_X2")
bad <- it[order(-it$RMSEA.S_X2), ]
head(bad, 10)

plot(model, type = "trace", empirical=TRUE,which.items = c(3,1))
