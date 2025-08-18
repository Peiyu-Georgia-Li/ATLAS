library(mirt)


#data <- read.csv("~/llmbenchmark/leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv")
data<-read.csv("~/llmbenchmark/cleaned_8b_model_train.csv")

# Drop constant columns (no variance)
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

# Optionally drop constant rows
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")

model <- mirt(data = clean_data[, -1], model = 1, itemtype = "3PL", technical = list(NCYCLES = 100000))
# Get dimensions of the cleaned data
#cat("Dimensions of cleaned data:", dim(data_matrix), "\n")


cat("===MODEL===")
print(model)

cat("====M2====")
M2(model, calcNull = TRUE)
#cat("===ITEMFIT===")
#itemfit(model,na.rm = TRUE)
it <- itemfit(model, fit_stats = "S_X2")
bad <- it[order(-it$RMSEA.S_X2), ]
head(bad, 10)
plot(model, type = "trace", which.items = c(1,6))  # 1= X0, 6= X5（按顺序号）

co <- coef(model, IRTpars = TRUE, simplify = TRUE)$items
co[c("X0","X2"), c("a","b","g","u")]   # 2PL/3PL/4PL 对应 a,b,c(g),u
mean(co[,"a"] < 0)                     # 若大多数都<0，可能是量表方向翻转

