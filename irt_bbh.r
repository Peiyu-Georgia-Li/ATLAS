options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("mirt", quietly = TRUE)) {
  install.packages("mirt")
}
if (!requireNamespace("WrightMap", quietly = TRUE)) {
  install.packages("WrightMap")
}

library(mirt)
library(WrightMap)

csv_file <- "leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv"
cat("Loading data from", csv_file, "...\n")

data <- read.csv(csv_file, header = TRUE, row.names = 1)

# Drop constant columns (no variance)
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

# Optionally drop constant rows
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")

data_matrix <- as.matrix(clean_data)

# Get dimensions of the cleaned data
cat("Dimensions of cleaned data:", dim(data_matrix), "\n")

# First select a subset of rows and columns
n_cols <- min(250, ncol(data_matrix))
n_rows <- min(100, nrow(data_matrix))
initial_subset <- data_matrix[1:n_rows, 1:n_cols]

# Check for columns with only one response category in the subset
single_category_cols <- apply(initial_subset, 2, function(x) length(unique(x)) == 1)
cat("Found", sum(single_category_cols), "columns with only one response category in the subset.\n")

# Remove columns with only one response category
reduced_data <- initial_subset[, !single_category_cols]
cat("Dimensions of reduced data after removing single-category columns:", dim(reduced_data), "\n")

# If we removed columns and now have fewer than 250, try to add more columns if available
if(sum(single_category_cols) > 0 && ncol(reduced_data) < 250 && ncol(data_matrix) > n_cols) {
  # How many more columns we need
  additional_cols_needed <- min(250 - ncol(reduced_data), ncol(data_matrix) - n_cols)
  
  if(additional_cols_needed > 0) {
    # Get additional columns
    additional_cols_start <- n_cols + 1
    additional_cols_end <- min(n_cols + additional_cols_needed, ncol(data_matrix))
    additional_cols <- data_matrix[1:n_rows, additional_cols_start:additional_cols_end]
    
    # Check these additional columns for single categories
    additional_single_cats <- apply(additional_cols, 2, function(x) length(unique(x)) == 1)
    valid_additional_cols <- additional_cols[, !additional_single_cats]
    
    if(ncol(valid_additional_cols) > 0) {
      # Add the valid additional columns to our reduced dataset
      reduced_data <- cbind(reduced_data, valid_additional_cols)
      cat("Added", ncol(valid_additional_cols), "additional columns.\n")
    }
  }
}

cat("Final dimensions of reduced data:", dim(reduced_data), "\n")

# Print a small sample to verify
sample_data <- reduced_data[1:5, 1:5]
cat("Sample of reduced data (5x5):\n")
print(sample_data)

# Fit IRT model on the reduced dataset
cat("\nFitting IRT model on reduced dataset...\n")
model <- mirt(reduced_data, 1, itemtype = "2PL")
summary(model)

# Extract fit indices
cat("\nAdditional Fit Indices:\n")
fit_indices <- c(
  logLik = extract.mirt(model, 'logLik'),
  df = extract.mirt(model, 'df'),
  AIC = extract.mirt(model, 'AIC'),
  BIC = extract.mirt(model, 'BIC'),
  SABIC = extract.mirt(model, 'SABIC'),
  HQ = extract.mirt(model, 'HQ'),
  G2 = extract.mirt(model, 'G2'),
  RMSEA = extract.mirt(model, 'RMSEA'),
  TLI = extract.mirt(model, 'TLI'),
  CFI = extract.mirt(model, 'CFI')
)
print(fit_indices)
