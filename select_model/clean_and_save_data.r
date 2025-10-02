# BENCHMARK<-"mmlu"
# BENCHMARK<-"truthfulqa"
# BENCHMARK<-"winogrande"
BENCHMARK<-"hellaswag"

# Load the RDS data
df <- readRDS(paste0("/store01/nchawla/pli9/llmbenchmark/metabench_data/benchmark-data/", BENCHMARK, "-preproc.rds"))

# Read the gaussian sampled models
gaussian_models <- read.csv(paste0("/store01/nchawla/pli9/metabench/scraping/gaussian_sampled_", BENCHMARK, "_models.csv"))
# gaussian_models <- read.csv("/store01/nchawla/pli9/metabench/scraping/gsm8k_models_filtered_300.csv")


# Get the model names from the gaussian sampled models file
model_names <- gaussian_models$model_name
# model_names <- gaussian_models$name

# Filter df$data to only include the models from gaussian_models
filtered_data <- df$data[rownames(df$data) %in% model_names, ]

# If any models from gaussian_models are missing in df$data, print a warning
missing_models <- model_names[!model_names %in% rownames(df$data)]
if (length(missing_models) > 0) {
  cat("Warning: The following models from gaussian_sampled_mmlu_models.csv were not found in the RDS data:\n")
  print(missing_models)
}

# Create a new dataframe with only 0s and 1s
# Each row is a model name and each column is an item
result_matrix <- filtered_data

# Save the result to a CSV file
write.csv(result_matrix, paste0("/store01/nchawla/pli9/llmbenchmark/select_model/gaussian_sampled_", BENCHMARK, "_response_matrix.csv"), row.names = TRUE)

cat("Process completed. Result saved to /store01/nchawla/pli9/llmbenchmark/select_model/gaussian_sampled_", BENCHMARK, "_response_matrix.csv\n")
cat("Number of models included:", nrow(result_matrix), "\n")
cat("Number of items:", ncol(result_matrix), "\n")
