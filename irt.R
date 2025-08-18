# Set a CRAN mirror first to avoid the "trying to use CRAN without setting a mirror" error
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install required packages if they are not already installed
if (!requireNamespace("mirt", quietly = TRUE)) {
  install.packages("mirt")
}
# if (!requireNamespace("WrightMap", quietly = TRUE)) {
#   install.packages("WrightMap")
# }

# Load the required libraries
library(mirt)
# library(WrightMap)

# Load the actual data from CSV file
csv_file <- "leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv"

cat("Loading data from", csv_file, "...\n")

# Read the CSV file
data <- read.csv(csv_file, header = TRUE, row.names = 1)

# Convert data frame to matrix (required for mirt)
data_matrix <- as.matrix(data)

# Display some basic info about the data
cat("Data dimensions:", dim(data_matrix), "\n")
cat("First few rows and columns of the data:\n")
print(data_matrix[1:5, 1:5])  # Show a small preview

# Check for missing data
na_count <- sum(is.na(data_matrix))
rows_with_na <- sum(apply(data_matrix, 1, function(x) any(is.na(x))))
cols_with_na <- sum(apply(data_matrix, 2, function(x) any(is.na(x))))
cat("\nMissing data check:\n")
cat("Number of NA values:", na_count, "\n")
cat("Number of rows with any NA:", rows_with_na, "\n")
cat("Number of columns with any NA:", cols_with_na, "\n")
cat("Note: Missing data will be handled with na.rm=TRUE in fit statistics calculations\n")

# Function to identify exact locations of missing values
find_missing_values <- function(data_mat) {
  # Find rows with missing values
  rows_with_na <- which(apply(data_mat, 1, function(x) any(is.na(x))))
  row_names <- rownames(data_mat)[rows_with_na]
  
  # For each row with NAs, find which columns have NAs
  missing_details <- list()
  for (i in rows_with_na) {
    row_name <- rownames(data_mat)[i]
    na_cols <- which(is.na(data_mat[i,]))
    missing_details[[row_name]] <- na_cols
  }
  
  # Print detailed report
  cat("\nDetailed Missing Value Report:\n")
  cat("===========================\n")
  
  if (length(missing_details) == 0) {
    cat("No missing values found.\n")
    return(invisible(NULL))
  }
  
  for (row_name in names(missing_details)) {
    na_cols <- missing_details[[row_name]]
    cat("Row '", row_name, "' has", length(na_cols), "missing values in columns:\n")
    
    # Print column indices in groups of 10 for readability
    col_groups <- split(na_cols, ceiling(seq_along(na_cols)/10))
    for (group in col_groups) {
      cat("  ", paste(group, collapse=", "), "\n")
    }
    cat("\n")
  }
  
  # Return the missing details invisibly for potential further use
  return(invisible(missing_details))
}

# Run the function to identify missing values
cat("\nLocating exact positions of missing values...\n")
missing_values_report <- find_missing_values(data_matrix)


# # Create a reduced dataset (100 rows x 250 columns)
# reduced_data <- data_matrix[1:250, 1:100]

# Fit a 3PL model using mirt
# 3PL includes discrimination (a), difficulty (b), and guessing (c) parameters
cat("Fitting 3PL IRT model...\n")
# Create control parameters for better convergence
control <- list(
  maxit = 1000,       # Increase max iterations from default 500
  TOL = 1e-5,         # Tighter convergence tolerance
  SEtol = 0.001,      # Standard error tolerance
  trace = FALSE        # Show iteration progress
)

# Fit the model with improved convergence settings
model <- mirt(data_matrix, 1, itemtype = "3PL", 
             technical = list(NCYCLES = 2000),  # Allow more MHRM cycles if needed
             control = control, 
             na.rm = TRUE
            )
cat("\n====PRINT MODEL====\n")
print(model)

# # # Add comprehensive model fit diagnostics
# cat("\n\n==== MODEL FIT DIAGNOSTICS ====\n")

# # # 1. Model fit statistics - Use try-catch to handle potential memory issues
# # cat("\n1. Overall Model Fit Statistics:\n")

# # fit_M2 <- M2(model, na.rm=TRUE) # Error: vector memory limit of 16.0 Gb reached, see mem.maxVSize()

# # # 2. Additional fit indices
# cat("\n2. Additional Fit Indices:\n")
# fit_indices <- c(
#   AIC = extract.mirt(model, 'AIC'),
#   BIC = extract.mirt(model, 'BIC'),
#   SABIC = extract.mirt(model, 'SABIC'),
#   HQ = extract.mirt(model, 'HQ'),
#   G2 = extract.mirt(model, 'G2'),
#   RMSEA = extract.mirt(model, 'RMSEA'),
#   TLI = extract.mirt(model, 'TLI'),
#   CFI = extract.mirt(model, 'CFI')
# )
# print(fit_indices)

# # # 3. Item fit statistics
# cat("\n3. Item Fit Statistics:\n")
# item_fit <- itemfit(model, na.rm=TRUE)
# print(item_fit)

# # # 4. Person fit statistics
# cat("\n4. Person Fit Statistics (first 10 persons):\n")
# # Calculate factor scores first and pass them to personfit
# theta_scores <- fscores(model, method = "EAP", full.scores = TRUE)
# person_fit <- personfit(model, Theta = theta_scores, na.rm = TRUE)
# print(head(person_fit, 10))

# # # 5. Standardized residuals
# # cat("\n5. Standardized Residuals (first 5x5):\n")
# # std_resid <- residuals(model, type = "standardized")
# # print(std_resid[1:5, 1:5])

# # # 6. Q3 statistics for local dependence
# # cat("\n6. Q3 Statistics for Local Dependence (first 5x5):\n")
# # Q3 <- residuals(model, type = "Q3")
# # print(Q3[1:5, 1:5])

# # 7. Compare with 2PL model
# cat("\n7. Comparing with 2PL Model:\n")
# model_2pl <- mirt(data_matrix, 1, itemtype = "2PL", verbose = FALSE)
# model_comparison <- anova(model_2pl, model)
# print(model_comparison)

# # Enhanced visualizations
# cat("\n\n==== ENHANCED VISUALIZATIONS ====\n")

# # Item parameter estimates
# cat("\nItem Parameters:\n")
# item_params <- coef(model, simplify = TRUE)$items
# print(item_params)



# # Save results to files
# benchmark_name <- "leaderboard_bbh_tracking_shuffled_objects_five_objects"
# output_params_file <- paste0("irt_item_parameters_", benchmark_name, ".csv")
# output_scores_file <- paste0("irt_person_scores_", benchmark_name, ".csv")
# output_fit_file <- paste0("irt_model_fit_", benchmark_name, ".csv")
# output_itemfit_file <- paste0("irt_item_fit_", benchmark_name, ".csv")

# write.csv(item_params, output_params_file, row.names = TRUE)
# write.csv(theta_scores, output_scores_file, row.names = FALSE)
# write.csv(person_fit, output_fit_file, row.names = FALSE)
# write.csv(item_fit, output_itemfit_file, row.names = TRUE)

# cat("\n3PL IRT analysis complete.\n")
# cat("Results saved to:\n")
# cat("- Item parameters:", output_params_file, "\n")
# cat("- Person scores:", output_scores_file, "\n")
# cat("- Model fit indices:", output_fit_file, "\n")
# cat("- Item fit statistics:", output_itemfit_file, "\n")

# # Interpretation guide
# cat("\n\n==== INTERPRETATION GUIDE ====\n")
# cat("
# Model Fit Interpretation:
# 1. RMSEA < 0.05 indicates good fit, 0.05-0.08 acceptable, > 0.08 poor fit
# 2. CFI/TLI > 0.95 indicates good fit, 0.90-0.95 acceptable
# 3. Lower AIC/BIC values indicate better fit when comparing models
# 4. For item fit, p-values > 0.05 suggest adequate fit
# 5. For person fit, infit/outfit values between 0.7 and 1.3 are ideal

# Item Parameters:
# - a: Discrimination (higher values mean the item better differentiates between abilities)
# - b: Difficulty (higher values mean more difficult items)
# - c: Guessing parameter (probability of correct answer by guessing)

# The Wright Map shows the distribution of person abilities (left) and item difficulties (right)
# on the same scale, allowing direct comparison.
# ")