# Set a CRAN mirror first to avoid the "trying to use CRAN without setting a mirror" error
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install required packages if they are not already installed
if (!requireNamespace("mirt", quietly = TRUE)) {
  install.packages("mirt")
}

# Load the mirt library
library(mirt)

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

# Fit a 3PL model using mirt
# 3PL includes discrimination (a), difficulty (b), and guessing (c) parameters
cat("Fitting 3PL IRT model...\n")
# Create control parameters for better convergence
control <- list(
  maxit = 1000,       # Increase max iterations from default 500
  TOL = 1e-5,         # Tighter convergence tolerance
  SEtol = 0.001,      # Standard error tolerance
  trace = TRUE        # Show iteration progress
)

# Fit the model with improved convergence settings
model <- mirt(data_matrix, 1, itemtype = "3PL", 
             verbose = TRUE,    # Show more details
             technical = list(NCYCLES = 2000),  # Allow more MHRM cycles if needed
             control = control)

# Display model summary
cat("\nModel Summary:\n")
summary(model)

# Item parameter estimates
cat("\nItem Parameters:\n")
item_params <- coef(model, simplify = TRUE)$items
print(item_params)

# Item characteristic curves
cat("\nPlotting Item Characteristic Curves for first 4 items...\n")
plot(model, type = "trace", which.items = 1:min(4, ncol(data_matrix)))

# Test information function
cat("\nPlotting Test Information Function...\n")
plot(model, type = "info")

# Person parameter estimates (theta scores)
cat("\nCalculating person ability scores...\n")
person_scores <- fscores(model)
cat("First few person scores:\n")
print(head(person_scores))
benchmark_name <- "leaderboard_bbh_tracking_shuffled_objects_five_objects"
# Save results to a file
output_params_file <- paste0("irt_item_parameters_", benchmark_name, ".csv")
output_scores_file <- paste0("irt_person_scores_", benchmark_name, ".csv")

write.csv(item_params, output_params_file, row.names = TRUE)
write.csv(person_scores, output_scores_file, row.names = FALSE)

cat("\n3PL IRT analysis complete.\n")
cat("Results saved to", output_params_file, "and", output_scores_file, "\n")