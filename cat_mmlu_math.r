# 1. Install and Load Required Packages

if (!require(catR)) install.packages("catR")

# Load libraries
library(catR)


# 2. Prepare Your Item Bank for catR
# Read and prepare item bank
FILE_PATH <- "mmlu_math_113_8b/irt_item_parameters_combined.csv"
prepare_item_bank <- function(file_path) {
  # Read your CSV
  items <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Ensure required columns exist
  required_cols <- c("X","a1","d","g","u")
  if (!all(required_cols %in% colnames(items))) {
    missing_cols <- setdiff(required_cols, colnames(items))
    stop(paste("Missing required columns in CSV:", paste(missing_cols, collapse = ", ")))
  }
  
  # Create empty matrix for catR parameters (a,b,c,d format)
  n_items <- nrow(items)
  item_bank <- matrix(nrow = n_items, ncol = 4)
  
  # Convert from (a1,d,g,u) to (a,b,c,d) format
  # a = absolute value of a1 (discrimination parameter)
  item_bank[,1] <- abs(items$a1) 
  
  # b = -d/a1 (difficulty parameter)
  item_bank[,2] <- -items$d / items$a1
  
  # c = g (guessing parameter)
  item_bank[,3] <- items$g
  
  # d = u (upper asymptote)
  item_bank[,4] <- items$u
  
  # Set column names to standard IRT notation
  colnames(item_bank) <- c("a", "b", "c", "d")
  
  # Set row names to item IDs
  rownames(item_bank) <- items$X
  
  # Ensure all parameters are numeric
  if (!all(is.numeric(item_bank))) {
    stop("Item parameters must be numeric")
  }
  
  # Print summary statistics of the converted parameters
  cat("Parameter ranges after conversion:\n")
  cat("a (discrimination): ", min(item_bank[,1]), "to", max(item_bank[,1]), "\n")
  cat("b (difficulty): ", min(item_bank[,2]), "to", max(item_bank[,2]), "\n")
  cat("c (guessing): ", min(item_bank[,3]), "to", max(item_bank[,3]), "\n")
  cat("d (upper asymptote): ", min(item_bank[,4]), "to", max(item_bank[,4]), "\n")
  
  # Store the item id separately
  item_info <- items[, c("X")]
  
  return(list(item_bank = item_bank, item_info = item_info))
}




# 3. Function to get score from different models

score_response <- function(model_name, item_id) {
  # Read the data once
  response_data <- read.csv("mmlu_pro/leaderboard_mmlu_pro_response_matrix_math.csv", stringsAsFactors = FALSE, check.names = FALSE)
  
  # The first column has an empty name but contains model names
  # Extract the first column (model names)
  model_names <- response_data[[1]]
  
  # Find the row index for the model
  row_idx <- which(model_names == model_name)
  
  # Remove X prefix from item_id if it exists
  item_number <- gsub("^X", "", item_id)
  
  # Check if we found the model and if the item_id exists in columns
  if (length(row_idx) > 0 && item_number %in% colnames(response_data)) {
    return(response_data[row_idx, item_number])
  } else {
    warning(paste("Could not find data for model", model_name, "and item", item_id, "/", item_number))
    return(NA)
  }
}

# # Debug the CSV structure
# print("Loading response matrix for debugging:")
# response_data <- read.csv("mmlu_pro/leaderboard_mmlu_pro_response_matrix_math.csv", stringsAsFactors = FALSE, check.names = FALSE)
# print("First few column names:")
# print(head(colnames(response_data)))
# print("Dimensions of the response matrix:")
# print(dim(response_data))
# print("First few model names:")
# print(head(response_data[[1]]))

# # Test the function with a valid item ID from the dataset
# print("Testing score_response function with a valid model and item ID:")
# valid_item_id <- colnames(response_data)[2]  # First column after index is a valid item ID
# valid_model <- response_data[[1]][1]  # First model in the dataset
# print(paste("Valid item ID:", valid_item_id))
# print(paste("Valid model name:", valid_model))
# print("Response value:")
# print(score_response(valid_model, valid_item_id))

# 4. CAT Implementation with catR
# Function to run CAT with catR and GenAI
run_cat <- function(item_bank, item_info, model_name,
                               start_theta = 0, 
                               min_items = 3, 
                               max_items = 8,
                               se_theta_stop = 0.1) {
  
  # Initialize test
  test <- list(
    administered = integer(0),  # Indices of administered items
    responses = numeric(0),     # Response vector (0/1)
    thetas = start_theta,       # Ability estimates
    SEs = NA,                   # Standard errors
    items_used = character(0)  # Item IDs
  )
  
  # CAT loop
  for (i in 1:max_items) {
    cat(sprintf("\n--- Item %d ---\n", i))
    
    tryCatch({
      # Select next item
      if (i == 1) {
        # First item: select based on start_theta
        next_item_pos <- which.min(abs(item_bank[,"b"] - start_theta))
      } else {
        # Subsequent items: use catR's selection
        next_item <- nextItem(item_bank, 
                              theta = tail(test$thetas, 1),
                              out = test$administered,
                              criterion = "MFI", 
                              method = "BM")
        next_item_pos <- next_item$item
      }
      
      # Get item details
      next_item_id <- rownames(item_bank)[next_item_pos]

      
      cat("Selected item:", next_item_id, "\n")
      
      # Score the response
      score <- score_response(model_name, next_item_id)

      cat("Score:", score, "Item id:", next_item_id, "\n")
      
      # Update test information
      test$administered <- c(test$administered, next_item_pos)
      test$responses <- c(test$responses, score)
      test$items_used <- c(test$items_used, next_item_id)

      
      # Estimate new theta
      est <- thetaEst(item_bank[test$administered, , drop = FALSE], 
                      test$responses, 
                      method = "BM")
      
      # Calculate standard error
      se <- semTheta(est, item_bank[test$administered, , drop = FALSE], test$responses)
      
      test$thetas <- c(test$thetas, est)
      test$SEs <- c(test$SEs, se)
      
      cat(sprintf("Current theta: %.2f, SE: %.3f, Running score: %d/%d\n", 
                  est, se, sum(test$responses), length(test$responses)))
      
      # Check stopping rules
      if (i >= min_items && !is.na(se) && se <= se_theta_stop) {
        cat(sprintf("\nStopping: SE (%.3f) <= %.3f\n", se, se_theta_stop))
        break
      }
      
    }, error = function(e) {
      cat("Error in CAT loop:", e$message, "\n")
      # If we have at least one response, continue with the next item
      if (length(test$responses) > 0) {
        cat("Continuing with next item...\n")
      } else {
        stop("Fatal error in CAT loop")
      }
    })
  }
  
  # Prepare final results
  final_theta <- tail(test$thetas, 1)
  final_se <- tail(test$SEs, 1)
  
  results <- data.frame(
    item_id = test$items_used,
    score = test$responses,
    theta = test$thetas[-1],
    se = test$SEs[-1],
    stringsAsFactors = FALSE
  )
  
  return(list(
    results = results,
    final_theta = final_theta,
    final_se = final_se,
    num_items = length(test$responses),
    thetas = test$thetas,
    SEs = test$SEs
  ))
}



###########################################
# FUNCTIONS: compare theta_full with theta_cat
###########################################
rmse_thetas <- function(theta_full, theta_cat) {
  # Calculate RMSE
  rmse <- sqrt(mean((theta_full - theta_cat)^2))
  cat("RMSE between theta_full and theta_cat: ", rmse, "\n")
  return(rmse)
}

correlation_thetas <- function(theta_full_list, theta_cat_list, plot_name="cat/correlation_thetas.png") {
  theta_full_vector <- unlist(theta_full_list)
  theta_cat_vector <- unlist(theta_cat_list)

  # Calculate correlation
  correlation <- cor(theta_full_vector, theta_cat_vector)
  cat("Correlation between theta_full and theta_cat: ", correlation, "\n")

  # png(plot_name, width=800, height=600)
  # # Scatter plot with identity line
  # plot(theta_full_vector, theta_cat_vector,
  #      main = paste("θ: Full vs CAT (r =", round(correlation, 3), ")"),
  #      xlab = "Full-test Theta",
  #      ylab = "CAT Theta",
  #      col = "blue", pch = 16)
  # abline(0, 1, col = "red", lty = 2)  # identity line
  # dev.off()
  # cat("\nPlots saved as ", plot_name, "\n")
  return(correlation)
}


# png("cat_theta_progression.png", width=800, height=600)
# plot(0:cat_results$num_items, cat_results$thetas, type = "b", 
#      xlab = "Item Number", ylab = "Theta Estimate",
#      main = "Ability Estimation During CAT")
# dev.off()

# cat("\nPlots saved as 'cat_theta_progression.png'\n")

#################################################
# MAIN
#################################################
theta_full_list <- list()
theta_cat_list <- list()
rmse_list <- list()
se_list <- list()
num_items_administered_list <- list()
time_list <- list()
theta_full_db <- read.csv("irt_person_scores_113_8b_math_WLE.csv")
se_theta_stop <- 0.15


# 1. Prepare your item bank
item_data <- prepare_item_bank(FILE_PATH)

# Run this to check your item bank structure
str(item_data$item_bank)
head(item_data$item_bank)
dim(item_data$item_bank)
###########################################
# CLEAN RESPONSE MATRIX
data <- read.csv("8b_leaderboard_mmlu_pro_response_matrix_math.csv")
# Get dimensions of the original data
cat("Dimensions of origial data:", dim(data), "\n")

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

# Get dimensions of the cleaned data
cat("Dimensions of cleaned data:", dim(clean_data), "\n")

###########################################
# GET MODEL
# Get first model from the response matrix to run the analysis
test_length <- dim(clean_data)[2]
test_length <- 3
# for (i in 1:dim(clean_data)[2]) {
for (i in 1:test_length) {
  cat("\n")
  cat("-----------------------------------------------\n")
  cat("Model", i, "\n")
  cat("-----------------------------------------------\n")
print("Loading response matrix to find model to test:")
test_model <- clean_data[[1]][i]  # First model in the dataset
print(paste("Using model", i, "for analysis:", test_model))

###########################################
# GET theta_full
theta_full <- theta_full_db[[2]][i] # First model theta in the dataset
print(theta_full)

###########################################
# 2. Run the CAT
start_time <- Sys.time()
set.seed(123)  # For reproducibility
cat_results <- run_cat(
  item_bank = item_data$item_bank,
  item_info = item_data$item_info,
  model_name = test_model,  
  start_theta = 0,      # Initial ability estimate
  min_items = 30,        # Minimum number of items
  max_items = 500,       # Maximum number of items
  se_theta_stop = se_theta_stop   # Stop when SE <= 0.3
)
end_time <- Sys.time()
time_taken <- end_time - start_time
duration_sec <- as.numeric(time_taken, units = "secs")
print("Duration in seconds:")
print(duration_sec)
###########################################
# Print item response sequence
cat("\nItem Response Sequence:\n")
for (i in 1:nrow(cat_results$results)) {
  cat(sprintf("Item %d: %s - Response: %d - Theta: %.2f (SE: %.3f)\n", 
              i, 
              cat_results$results$item_id[i],
              cat_results$results$score[i],
              cat_results$results$theta[i],
              cat_results$results$se[i]))
}
###########################################
# 4. View results
print("CAT Results Summary:")
print(cat_results$results)
cat("\nFinal theta estimate:", cat_results$final_theta, "\n")
cat("Standard Error:", cat_results$final_se, "\n")
cat("Number of items administered:", cat_results$num_items, "\n")



##########################################
# 5. Compare theta_full with theta_cat

rmse<-rmse_thetas(cat_results$final_theta, theta_full)

###########################################
# 6. Save results
theta_full_list <- append(theta_full_list, theta_full)
theta_cat_list <- append(theta_cat_list, cat_results$final_theta)
rmse_list <- append(rmse_list, rmse)
se_list <- append(se_list, cat_results$final_se)
num_items_administered_list <- append(num_items_administered_list, cat_results$num_items)
time_list <- append(time_list, duration_sec)
###########################################
# 7. Plot theta progression
# Save as PNG for easier viewing
# png("cat_theta_progression.png", width=800, height=600)
# plot(0:cat_results$num_items, cat_results$thetas, type = "b", 
#      xlab = "Item Number", ylab = "Theta Estimate",
#      main = "Ability Estimation During CAT")
# dev.off()

# cat("\nPlots saved as 'cat_theta_progression.png'\n")
}

correlation_thetas(theta_full_list, theta_cat_list)
# print("----------------")
# print(theta_cat_list)
# print("----------------")
# print(theta_full_list)
# print("----------------")
# Get model names from the first column
model_names <- theta_full_db[[1]][1:test_length]
head(model_names)
# print(model_names)
# print("----------------")
# print(unlist(theta_cat_list))
# print("----------------")
# print(unlist(theta_full_list))
# Create data frame with model names and theta values
theta_df <- data.frame(Model_Name = model_names, Theta_CAT = unlist(theta_cat_list), Theta_WLE = unlist(theta_full_list), RMSE = unlist(rmse_list), SE = unlist(se_list), Num_Items = unlist(num_items_administered_list), Time_Taken_Sec = unlist(time_list))

# Save results
output_file <- paste0("cat/irt_person_scores_113_8b_math_CAT_", se_theta_stop, ".csv")
write.csv(theta_df, output_file, row.names = FALSE)
print(paste("Saved person scores to", output_file))
  