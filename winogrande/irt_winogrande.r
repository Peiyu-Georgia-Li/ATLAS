# usage: 
# for i in 105 209 313 417 521 626 731 836 941 1046; do
#   Rscript irt_winogrande.r $i
# done
library(mirt)

args <- commandArgs(trailingOnly = TRUE)
index_name <- args[1]
i <- as.numeric(args[1])
print(i)

clean_data<-read.csv("../data/clean_response_matrix_winogrande.csv")


# chunk sizes
sizes <- c(104, 104, 104, 104, 104, 105, 105,105, 105, 105)
ends  <- cumsum(sizes) + 1  # since starting col is 2
# find which chunk this i belongs to
chunk_idx <- which(ends == i)
if (chunk_idx == 1) {
  start_col <- 2
} else {
  start_col <- ends[chunk_idx - 1] + 1
}
dat <- clean_data[, start_col:i]

model <- mirt(dat, model = 1, method="EM", itemtype = "3PL", technical = list(NCYCLES = 100000))

cat("===MODEL===\n")
print(model)

cat("====M2====\n")
m2<-M2(model)
print(m2)
cat("\nTheta Scores:\n")
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE, full.scores.SE = TRUE, quadpts = 61)
cat("\nItem Parameters:\n")
item_params <- coef(model, simplify = TRUE)$items
print(item_params)


# # Save results to files

path <- "./"

if (!dir.exists(path)) dir.create(path)
output_scores_file <- paste0(path, "irt_person_scores_", index_name, ".csv")
output_params_file <- paste0(path,"irt_item_parameters_", index_name, ".csv")
output_m2_file<- paste0(path, "m2_", index_name, ".csv")
write.csv(theta_scores, output_scores_file, row.names = FALSE)
write.csv(item_params, output_params_file, row.names = TRUE)
write.csv(m2, output_m2_file, row.names = TRUE)


