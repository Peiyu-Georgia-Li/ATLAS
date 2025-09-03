library(mirt)

args <- commandArgs(trailingOnly = TRUE)
benchmark_name <- args[1]
i <- as.numeric(args[1])
print(i)
#data<-read.csv("~/llmbenchmark/leaderboard_mmlu_pro_response_matrix.csv")
#data <- read.csv("~/llmbenchmark/cleaned_leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv")
#data<-read.csv("/Users/lipeiyu/llmbenchmark/cleaned_8b_model_train.csv")
#data<-read.csv("~/llmbenchmark/positive_items_for_mirt.csv")
data <- read.csv("/Users/lipeiyu/llmbenchmark/mmlu_pro/leaderboard_mmlu_pro_response_matrix_math.csv")


data_clean <- na.omit(data)              # remove NA rows
data <- data_clean[, colSums(is.na(data_clean)) == 0]  # remove NA columns

# Drop constant columns (no variance)
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

# Optionally drop constant rows
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")

#dat<-clean_data[, (i-9):i]
dat<-clean_data[, (i-99):i]
#dat<-clean_data[, 2:101]

#head(dat)
model <- mirt(dat, model = 2, method="EM", itemtype = "3PL", technical = list(NCYCLES = 100000))
# Get dimensions of the cleaned data
#cat("Dimensions of cleaned data:", dim(data_matrix), "\n")


cat("===MODEL===\n")
print(model)

cat("====M2====\n")
m2<-M2(model)
print(m2)
summary(model, rotate = "oblimin", suppress = 0.20)
cat("===Q3===\n")
# Suppose your Q3 matrix is called q3mat
q3mat <- residuals(model, type = "Q3")
print(q3mat)
# Get upper triangle only (to avoid duplicates)
q3_upper <- q3mat
q3_upper[lower.tri(q3_upper, diag = TRUE)] <- NA

# Find pairs with Q3 > 0.20
threshold <- 0.20
bad_pairs <- which(q3_upper > threshold, arr.ind = TRUE)

# Format results nicely
results <- data.frame(
  Item1 = rownames(q3_upper)[bad_pairs[,1]],
  Item2 = colnames(q3_upper)[bad_pairs[,2]],
  Q3    = q3_upper[bad_pairs]
)

# Sort by strongest dependence
q3 <- results[order(-results$Q3), ]
print(q3)

# # Define main factor F1 (all items load here)
# # Define testlet factors (T1 for X5,X9; T2 for X8,X10)
# items_data <- data[, 2:12]
# model_spec <- 'F1 = 1-11\nCONSTRAIN = (1-11, a1)\nT1 = 5, 9\nT2 = 8, 10\nT3 = 1, 4'
# fit_testlet <- mirt(items_data, model_spec, itemtype = '3PL', technical = list(NCYCLES = 10000))
# print(fit_testlet)
# M2(fit_testlet)

# summary(fit_testlet)
# anova(model, fit_testlet)
############################################################
#cat("===ITEMFIT===\n")
#itemfit(model,na.rm = TRUE)
item_fit <- itemfit(model, fit_stats = "S_X2")
bad <- item_fit[order(-item_fit$RMSEA.S_X2), ]
head(bad, 10)
plot(model, type = "trace", which.items = c(1,6))  # 1= X0, 6= X5（按顺序号）

# co <- coef(model, IRTpars = TRUE, simplify = TRUE)$items
# co[c("X0","X2"), c("a","b","g","u")]   # 2PL/3PL/4PL 对应 a,b,c(g),u
# mean(co[,"a"] < 0)                     # 若大多数都<0，可能是量表方向翻转


# # 4. Person fit statistics
cat("\n4. Person Fit Statistics (first 10 persons):\n")
# Calculate factor scores first and pass them to personfit
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE)
person_fit <- personfit(model, Theta = theta_scores, na.rm = TRUE)
print(head(person_fit, 10))
# Item parameter estimates
cat("\nItem Parameters:\n")
item_params <- coef(model, simplify = TRUE)$items
print(item_params)

# # Save results to files
path <- "~/llmbenchmark/mmlu_math_2/"

output_params_file <- paste0(path,"irt_item_parameters_", benchmark_name, ".csv")
output_scores_file <- paste0(path, "irt_person_scores_", benchmark_name, ".csv")
output_fit_file <- paste0(path, "irt_model_fit_", benchmark_name, ".csv")
output_itemfit_file <- paste0(path, "irt_item_fit_", benchmark_name, ".csv")
output_m2_file<- paste0(path, "m2_", benchmark_name, ".csv")
output_q3_file<- paste0(path, "q3_", benchmark_name, ".csv")
write.csv(item_params, output_params_file, row.names = TRUE)
write.csv(theta_scores, output_scores_file, row.names = FALSE)
write.csv(person_fit, output_fit_file, row.names = FALSE)
write.csv(bad, output_itemfit_file, row.names = TRUE)
write.csv(m2, output_m2_file, row.names = TRUE)
write.csv(q3, output_q3_file, row.names = TRUE)

