# 1. Install and Load Required Packages

if (!require(catR)) install.packages("catR")

# Load libraries
library(catR)


# 2. Prepare Your Item Bank for catR
# Read and prepare item bank
DIRECTORY <- "cat_gsm8k_random/"#⚠️
FILE_PATH <- "irt_item_parameters_combined.csv"
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
  item_bank[,1] <- items$a1
  
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
  response_data <- read.csv("../select_model/gaussian_sampled_gsm8k_response_matrix.csv", stringsAsFactors = FALSE, check.names = FALSE) #⚠️

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
    #   if (i == 1) {
    #     # First item: select based on start_theta
    #     next_item_pos <- which.min(abs(item_bank[,"b"] - start_theta))
    #   } else {
    #     # Subsequent items: use catR's selection
    #     next_item <- nextItem(item_bank, 
    #                           theta = tail(test$thetas, 1),
    #                           out = test$administered,
    #                           criterion = "MFI", 
    #                           method = "BM")
    #     next_item_pos <- next_item$item
    #   }

        # Select next item
    #   if (i == 1) {
    #     # First item: use randomesque selection from items near start_theta
    #     next_item <- nextItem(item_bank, 
    #                         theta = start_theta,
    #                         out = test$administered,
    #                         criterion = "MFI", 
    #                         method = "BM",
    #                         randomesque = 5)  # Randomly select from 5 most informative items
    #     next_item_pos <- next_item$item
    if (i == 1) {
  # First item: select randomly from items with difficulty near start_theta
  # Get items with difficulty within a certain range of start_theta
  difficulty_range <- 0.5  # Adjust this value to control diversity
  candidate_items <- which(abs(item_bank[,"b"] - start_theta) < difficulty_range)
  
  # If no items within range, fall back to closest item
  if(length(candidate_items) == 0) {
    next_item_pos <- which.min(abs(item_bank[,"b"] - start_theta))
  } else {
    # Randomly select from candidates
    next_item_pos <- sample(candidate_items, 1)
  }
      } else {
        # Subsequent items: use catR's selection with randomesque
        next_item <- nextItem(item_bank, 
                              theta = tail(test$thetas, 1),
                              out = test$administered,
                              criterion = "MFI", 
                              method = "EAP", #⚠️BM
                              randomesque = 5)  # Randomly select from 5 most informative items
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
                      method = "EAP")#⚠️BM
      
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
  
  # Save selected items to a CSV file
  items_df <- data.frame(
    item_id = test$items_used,
    order = 1:length(test$items_used),
    score = test$responses,
    theta = test$thetas[-1],
    se = test$SEs[-1],
    stringsAsFactors = FALSE
  )
  
  # Create directory if it doesn't exist
  dir.create(paste0(DIRECTORY, "selected_items_", se_theta_stop), recursive = TRUE, showWarnings = FALSE)
  
  # Generate filename with model name (cleaned to be safe for filenames)
  safe_model_name <- gsub("/|\\\\|:|\\*|\\?|\"|<|>|\\||\\s", "_", model_name)
  items_file <- paste0(DIRECTORY, "selected_items_", se_theta_stop, "/", safe_model_name, "_items.csv")
  
  # Save to CSV
  write.csv(items_df, items_file, row.names = FALSE)
  cat("Selected items saved to", items_file, "\n")
  
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

correlation_thetas <- function(theta_full_list, theta_cat_list, plot_name=paste0(DIRECTORY, "correlation_thetas_", se_theta_stop, ".png")) {
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
theta_full_db <- read.csv("irt_person_scores_WLE_SE.csv")
se_theta_stop <- 0.3 ##⚠️


# 1. Prepare your item bank
item_data <- prepare_item_bank(FILE_PATH)

# Run this to check your item bank structure
str(item_data$item_bank)
head(item_data$item_bank)
dim(item_data$item_bank)
###########################################
# CLEAN RESPONSE MATRIX
data <- read.csv("../select_model/gaussian_sampled_gsm8k_response_matrix.csv")##⚠️
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
test_length <- dim(clean_data)[1]
# test_length <- 2#⚠️
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
theta_df <- data.frame(Model_Name = model_names, Theta_CAT = unlist(theta_cat_list), Theta_WLE = unlist(theta_full_list), RMSE_single = unlist(rmse_list), SE = unlist(se_list), Num_Items = unlist(num_items_administered_list), Time_Taken_Sec = unlist(time_list))

# Save results
output_file <- paste0(DIRECTORY, "irt_person_scores_112_8b_math_CAT_", se_theta_stop, ".csv")
write.csv(theta_df, output_file, row.names = FALSE)
print(paste("Saved person scores to", output_file))
###########################################
# Calculate and report test overlap rate
###########################################

# Function to calculate test overlap rate between two item sets
calculate_overlap_rate <- function(items1, items2) {
  # Find common items
  common_items <- intersect(items1, items2)
  
  # Calculate overlap rate (percentage of items in common)
  overlap_rate <- length(common_items) / max(length(items1), length(items2))
  
  return(list(
    overlap_rate = overlap_rate,
    common_items = common_items,
    n_common = length(common_items)
  ))
}

# Function to calculate all pairwise overlap rates
calculate_all_overlaps <- function() {
  # Get all item files
  item_files <- list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$", full.names=TRUE)
  
  if(length(item_files) < 2) {
    cat("Need at least 2 item sets to calculate overlap\n")
    return(NULL)
  }
  
  # Read all item sets
  item_sets <- lapply(item_files, function(f) {
    df <- read.csv(f)
    return(df$item_id)
  })
  
  # Get model names from filenames
  model_names <- basename(item_files)
  model_names <- sub("_items\\.csv$", "", model_names)
  
  # Calculate all pairwise overlaps
  n_models <- length(item_sets)
  overlap_matrix <- matrix(0, nrow=n_models, ncol=n_models)
  
  for(i in 1:(n_models-1)) {
    for(j in (i+1):n_models) {
      overlap <- calculate_overlap_rate(item_sets[[i]], item_sets[[j]])
      overlap_matrix[i,j] <- overlap$overlap_rate
      overlap_matrix[j,i] <- overlap$overlap_rate
      
      cat("Overlap between", model_names[i], "and", model_names[j], ":", 
          round(overlap$overlap_rate * 100, 2), "%\n")
      cat("  Common items:", overlap$n_common, "\n")
    }
  }
  
  # Set row and column names
  rownames(overlap_matrix) <- model_names
  colnames(overlap_matrix) <- model_names
  
  # Calculate average overlap rate
  avg_overlap <- mean(overlap_matrix[upper.tri(overlap_matrix)])
  cat("\nAverage overlap rate:", round(avg_overlap * 100, 2), "%\n")
  
  # Save overlap matrix to CSV
  overlap_df <- as.data.frame(overlap_matrix)
  write.csv(overlap_df, paste0(DIRECTORY, "item_overlap_matrix_", se_theta_stop, ".csv"))
  cat("Overlap matrix saved to ", paste0(DIRECTORY, "item_overlap_matrix_", se_theta_stop, ".csv"), "\n")
  
  # Create heatmap visualization of overlap matrix
  if (requireNamespace("ggplot2", quietly = TRUE) && 
      requireNamespace("reshape2", quietly = TRUE)) {
    library(ggplot2)
    library(reshape2)
    
    # Prepare data for ggplot
    overlap_melt <- melt(overlap_matrix)
    names(overlap_melt) <- c("Model1", "Model2", "Overlap")
    
    # Create heatmap
    p <- ggplot(overlap_melt, aes(Model1, Model2, fill = Overlap)) +
      geom_tile(color = "white") +
      scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, 1)) +
      theme_minimal() +
      labs(title = "Item Overlap Rate Between Models", 
           fill = "Overlap Rate") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(hjust = 0.5))
    
    # Save plot
    ggsave(paste0(DIRECTORY, "item_overlap_heatmap_", se_theta_stop, ".png"), p, width = 8, height = 6)
    cat("Heatmap visualization saved to ", paste0(DIRECTORY, "item_overlap_heatmap_", se_theta_stop, ".png\n"))
  } else {
    cat("Note: ggplot2 and reshape2 packages are required for creating visualizations.\n")
    cat("Install them using: install.packages(c('ggplot2', 'reshape2'))\n")
  }
  
  return(overlap_matrix)
}

# Call the function to calculate overlaps if selected items exist
cat("\n---------- Item Overlap Analysis ----------\n")
if(dir.exists(paste0(DIRECTORY, "selected_items_", se_theta_stop)) && 
   length(list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$")) > 0) {
  overlap_results <- calculate_all_overlaps()
} else {
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run CAT first to generate item selection data.\n")
  overlap_results <- NULL
}

###########################################
# Analyze item selection frequency
###########################################

analyze_item_frequency <- function() {
  # Get all item files
  item_files <- list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$", full.names=TRUE)
  
  if(length(item_files) < 1) {
    cat("No item sets found\n")
    return(NULL)
  }
  
  # Read all item sets
  item_sets <- lapply(item_files, function(f) {
    df <- read.csv(f)
    return(df$item_id)
  })
  
  # Get model names from filenames
  model_names <- basename(item_files)
  model_names <- sub("_items\\.csv$", "", model_names)
  
  # Combine all items
  all_items <- unlist(item_sets)
  
  # Count frequency of each item
  item_freq <- table(all_items)
  
  # Convert to data frame
  item_freq_df <- data.frame(
    item_id = names(item_freq),
    frequency = as.numeric(item_freq),
    percentage = 100 * as.numeric(item_freq) / length(model_names)
  )
  
  # Sort by frequency
  item_freq_df <- item_freq_df[order(-item_freq_df$frequency),]
  
  # Save to CSV
  write.csv(item_freq_df, paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv"), row.names = FALSE)
  
  # Print top items
  cat("\nTop 10 most frequently selected items:\n")
  for(i in 1:min(10, nrow(item_freq_df))) {
    cat(sprintf("%s: %d models (%.1f%%)\n", 
                item_freq_df$item_id[i], 
                item_freq_df$frequency[i],
                item_freq_df$percentage[i]))
  }
  
  cat("\nItem selection frequency saved to ", paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv\n"))
  
  # Create plots if packages are available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    library(ggplot2)
    
    # Plot frequency distribution
    p <- ggplot(item_freq_df, aes(x = reorder(item_id, -frequency), y = frequency)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      theme_minimal() +
      labs(title = "Item Selection Frequency", 
           x = "Item ID", 
           y = "Number of Models") +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6))
    
    # Save plot
    ggsave(paste0(DIRECTORY, "item_frequency_plot_", se_theta_stop, ".png"), p, width = 10, height = 6)
    cat("Item frequency plot saved to ", paste0(DIRECTORY, "item_frequency_plot_", se_theta_stop, ".png\n"))
  }
  
  return(item_freq_df)
}

# Analyze item frequency if selected items exist
cat("\n---------- Item Selection Frequency Analysis ----------\n")
if(dir.exists(paste0(DIRECTORY, "selected_items_", se_theta_stop)) && 
   length(list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$")) > 0) {
  item_freq_results <- analyze_item_frequency()
} else {
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run CAT first to generate item selection data.\n")
  item_freq_results <- NULL
}

###########################################
# Analyze item selection positions
###########################################

analyze_item_positions <- function() {
  # Get all item files
  item_files <- list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$", full.names=TRUE)
  
  if(length(item_files) < 1) {
    cat("No item sets found\n")
    return(NULL)
  }
  
  # Read all item sets with position info
  item_position_data <- list()
  
  for(f in item_files) {
    df <- read.csv(f)
    model_name <- sub("_items\\.csv$", "", basename(f))
    
    for(i in 1:nrow(df)) {
      item_id <- df$item_id[i]
      position <- df$order[i]
      
      # Add to data collection
      if(!item_id %in% names(item_position_data)) {
        item_position_data[[item_id]] <- list(positions = position, models = model_name)
      } else {
        item_position_data[[item_id]]$positions <- c(item_position_data[[item_id]]$positions, position)
        item_position_data[[item_id]]$models <- c(item_position_data[[item_id]]$models, model_name)
      }
    }
  }
  
  # Create summary data frame
  position_summary <- data.frame(
    item_id = character(),
    frequency = integer(),
    mean_position = numeric(),
    min_position = integer(),
    max_position = integer(),
    sd_position = numeric(),
    stringsAsFactors = FALSE
  )
  
  for(item_id in names(item_position_data)) {
    positions <- item_position_data[[item_id]]$positions
    
    position_summary <- rbind(position_summary, data.frame(
      item_id = item_id,
      frequency = length(positions),
      mean_position = mean(positions),
      min_position = min(positions),
      max_position = max(positions),
      sd_position = if(length(positions) > 1) sd(positions) else NA,
      stringsAsFactors = FALSE
    ))
  }
  
  # Sort by frequency and then mean position
  position_summary <- position_summary[order(-position_summary$frequency, position_summary$mean_position),]
  # Save to CSV
  write.csv(position_summary, paste0(DIRECTORY, "item_position_analysis_", se_theta_stop, ".csv"), row.names = FALSE)
  
  # Print summary for top items
  cat("\nPosition analysis for most common items:\n")
  top_items <- head(position_summary, 10)
  for(i in 1:nrow(top_items)) {
    cat(sprintf("%s: Used in %d models, avg position %.1f (range: %d-%d)\n", 
                top_items$item_id[i], 
                top_items$frequency[i],
                top_items$mean_position[i],
                top_items$min_position[i],
                top_items$max_position[i]))
  }
  
  cat("\nItem position analysis saved to ",paste0(DIRECTORY, "item_position_analysis_", se_theta_stop, ".csv\n"))
  
  # Identify common initial items
  initial_items <- position_summary[position_summary$min_position == 1,]
  initial_items <- initial_items[order(-initial_items$frequency),]
  
  cat("\nCommon first items across models:\n")
  for(i in 1:min(5, nrow(initial_items))) {
    cat(sprintf("%s: First item in %d/%d models (%.1f%%)\n", 
                initial_items$item_id[i], 
                sum(item_position_data[[initial_items$item_id[i]]]$positions == 1),
                initial_items$frequency[i],
                100 * sum(item_position_data[[initial_items$item_id[i]]]$positions == 1) / length(item_files)))
  }
  
  return(position_summary)
}

# Analyze item positions if selected items exist
cat("\n---------- Item Position Analysis ----------\n")
if(dir.exists(paste0(DIRECTORY, "selected_items_", se_theta_stop)) && 
   length(list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$")) > 0) {
  position_results <- analyze_item_positions()
} else {
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run CAT first to generate item selection data.\n")
  position_results <- NULL
}

###########################################
# Generate comprehensive test overlap report
###########################################

generate_test_overlap_report <- function(se_theta_stop = NA) {
  # Create report file name with timestamp
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  report_filename <- paste0(DIRECTORY, "test_overlap_report_", se_theta_stop, "_", timestamp, ".txt")
  
  # Open report file
  sink(report_filename)
  
  cat("=============================================\n")
  cat("     COMPUTERIZED ADAPTIVE TESTING (CAT)    \n")
  cat("           TEST OVERLAP ANALYSIS            \n")
  cat("=============================================\n\n")
  
  cat("Report generated on:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  
  if(!is.na(se_theta_stop)) {
    cat("CAT stopping rule: SE(theta) ≤", se_theta_stop, "\n\n")
  }
  
  # Get item files and model count
  item_files <- list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$", full.names=TRUE)
  model_count <- length(item_files)
  
  cat("Number of models analyzed:", model_count, "\n\n")
  
  if(model_count < 2) {
    cat("Insufficient data for overlap analysis (at least 2 models required)\n")
    sink()
    return(report_filename)
  }
  
  # Basic test statistics
  all_item_counts <- numeric()
  for(f in item_files) {
    df <- read.csv(f)
    all_item_counts <- c(all_item_counts, nrow(df))
  }
  
  cat("Test length statistics:\n")
  cat("  Mean number of items:   ", round(mean(all_item_counts), 2), "\n")
  cat("  Median number of items: ", median(all_item_counts), "\n")
  cat("  Min number of items:    ", min(all_item_counts), "\n")
  cat("  Max number of items:    ", max(all_item_counts), "\n\n")
  
  # Overall overlap rate
  if(file.exists(paste0(DIRECTORY, "item_overlap_matrix_", se_theta_stop, ".csv"))) {
    overlap_matrix <- read.csv(paste0(DIRECTORY, "item_overlap_matrix_", se_theta_stop, ".csv"), row.names=1)
    avg_overlap <- mean(as.matrix(overlap_matrix)[upper.tri(as.matrix(overlap_matrix))])
    cat("Average test overlap rate: ", round(avg_overlap * 100, 2), "%\n\n")
  }
  
  # Frequently used items
  if(file.exists(paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv"))) {
    item_freq <- read.csv(paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv"))
    item_freq <- item_freq[order(-item_freq$frequency),]
    
    cat("Most frequently selected items:\n")
    for(i in 1:min(10, nrow(item_freq))) {
      cat(sprintf("  %s: %d models (%.1f%%)\n", 
                  item_freq$item_id[i], 
                  item_freq$frequency[i],
                  item_freq$percentage[i]))
    }
    cat("\n")
    
    # Report items used in >50% of tests
    high_freq_items <- item_freq[item_freq$percentage > 50,]
    cat("Number of items used in >50% of tests: ", nrow(high_freq_items), "\n")
    cat("Number of items used in >75% of tests: ", sum(item_freq$percentage > 75), "\n\n")
  }
  
  # Common first items
  if(file.exists(paste0(DIRECTORY, "item_position_analysis_", se_theta_stop, ".csv"))) {
    position_data <- read.csv(paste0(DIRECTORY, "item_position_analysis_", se_theta_stop, ".csv"))
    first_items <- position_data[position_data$min_position == 1,]
    first_items <- first_items[order(-first_items$frequency),]
    
    cat("Common first items across tests:\n")
    for(i in 1:min(5, nrow(first_items))) {
      cat(sprintf("  %s: Used in %d models, avg position %.1f\n", 
                  first_items$item_id[i], 
                  first_items$frequency[i],
                  first_items$mean_position[i]))
    }
    cat("\n")
  }
  
  # Item bank utilization
  if(exists("item_bank") && is.data.frame(item_bank)) {
    total_items <- nrow(item_bank)
    
    if(file.exists(paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv"))) {
      item_freq <- read.csv(paste0(DIRECTORY, "item_selection_frequency_", se_theta_stop, ".csv"))
      used_items <- nrow(item_freq)
      
      cat("Item bank utilization:\n")
      cat("  Total items in bank:   ", total_items, "\n")
      cat("  Items used at least once:", used_items, "(")
      cat(round(100 * used_items / total_items, 1), "%)\n")
      cat("  Items never selected:  ", total_items - used_items, "(")
      cat(round(100 * (total_items - used_items) / total_items, 1), "%)\n\n")
    }
  }
  
  cat("=============================================\n")
  cat("              END OF REPORT                \n")
  cat("=============================================\n")
  
  # Close report file
  sink()
  
  cat("Test overlap report saved to", report_filename, "\n")
  return(report_filename)
}

# Generate comprehensive report if selected items exist
cat("\n---------- Generating Comprehensive Test Overlap Report ----------\n")
if(dir.exists(paste0(DIRECTORY, "selected_items_", se_theta_stop)) && 
   length(list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$")) > 0) {
  overlap_report <- generate_test_overlap_report(se_theta_stop)
} else {
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run CAT first to generate item selection data.\n")
  overlap_report <- NULL
}

  