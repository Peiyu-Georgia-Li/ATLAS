# usage: Rscript atlas_random.r --se_theta_stop=0.1 --test_length=100 (if test_length is not specified, it will run on all models)
# Install and Load Required Packages

if (!require(catR)) install.packages("catR")
library(catR)

# Parse CLI arguments for se_theta_stop and test_length
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (a in args) {
    if (grepl("^--se_theta_stop=", a)) {
      out$se_theta_stop <- as.numeric(sub("^--se_theta_stop=", "", a))
    } else if (grepl("^--test_length=", a)) {
      out$test_length <- as.integer(sub("^--test_length=", "", a))
    }
  }
  return(out)
}
.cli_args <- parse_args()
if (!is.null(.cli_args$se_theta_stop) && !is.na(.cli_args$se_theta_stop)) {
  se_theta_stop <- .cli_args$se_theta_stop
} else {
  se_theta_stop <- 0.2
}

# Prepare item bank
# Read and prepare item bank
DIRECTORY <- "atlas_hellaswag_random/"
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
  
  # Create empty matrix for parameters (a,b,c,d format)
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




# Function to get score from different models

score_response <- function(model_name, item_id) {
  # Read the data once
  response_data <- read.csv("../data/clean_response_matrix_hellaswag.csv", stringsAsFactors = FALSE, check.names = FALSE) #⚠️

  # The first column has an empty name but contains model names
  # Extract the first column (model names)
  model_names <- response_data[[1]]

  # Find the row index for the model
  row_idx <- which(model_names == model_name)
  
  item_number <- item_id
  
  # Check if we found the model and if the item_id exists in columns
  if (length(row_idx) > 0 && item_number %in% colnames(response_data)) {
    return(response_data[row_idx, item_number])
  } else {
    warning(paste("Could not find data for model", model_name, "and item", item_id, "/", item_number))
    return(NA)
  }
}


# Function to run ATLAS 
run_atlas <- function(item_bank, item_info, model_name,
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
  
  # ATLAS loop
  for (i in 1:max_items) {
    cat(sprintf("\n--- Item %d ---\n", i))
    
    tryCatch({
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
        # Subsequent items: selection with randomesque
        next_item <- nextItem(item_bank, 
                              theta = tail(test$thetas, 1),
                              out = test$administered,
                              criterion = "MFI", 
                              method = "EAP", 
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
                      method = "EAP")
      
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
      cat("Error in ATLAS loop:", e$message, "\n")
      # If we have at least one response, continue with the next item
      if (length(test$responses) > 0) {
        cat("Continuing with next item...\n")
      } else {
        stop("Fatal error in ATLAS loop")
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


#################################################
# MAIN
#################################################
theta_full_list <- list()
theta_atlas_list <- list()
se_list <- list()
num_items_administered_list <- list()
time_list <- list()
theta_full_db <- read.csv("irt_person_scores_WLE_SE.csv")
# se_theta_stop is taken from CLI args parsed above; defaulted to 0.2 if not provided


# Prepare item bank
item_data <- prepare_item_bank(FILE_PATH)

# Check item bank structure
str(item_data$item_bank)
head(item_data$item_bank)
dim(item_data$item_bank)
###########################################
# CLEAN RESPONSE MATRIX
clean_data <- read.csv("../data/clean_response_matrix_winogrande.csv")

###########################################
# GET MODEL
# Determine number of models to run (can be overridden via --test_length)
test_length <- nrow(clean_data)
if (!is.null(.cli_args$test_length) && !is.na(.cli_args$test_length)) {
  # Clamp to valid range
  test_length <- max(1, min(.cli_args$test_length, nrow(clean_data)))
}
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
# Run ATLAS
start_time <- Sys.time()
set.seed(123)  # For reproducibility
atlas_results <- run_atlas(
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
for (i in 1:nrow(atlas_results$results)) {
  cat(sprintf("Item %d: %s - Response: %d - Theta: %.2f (SE: %.3f)\n", 
              i, 
              atlas_results$results$item_id[i],
              atlas_results$results$score[i],
              atlas_results$results$theta[i],
              atlas_results$results$se[i]))
}
###########################################
# Print results
print("ATLAS Results Summary:")
print(atlas_results$results)
cat("\nFinal theta estimate:", atlas_results$final_theta, "\n")
cat("Standard Error:", atlas_results$final_se, "\n")
cat("Number of items administered:", atlas_results$num_items, "\n")

##########################################
# Save results
theta_full_list <- append(theta_full_list, theta_full)
theta_atlas_list <- append(theta_atlas_list, atlas_results$final_theta)
se_list <- append(se_list, atlas_results$final_se)
num_items_administered_list <- append(num_items_administered_list, atlas_results$num_items)
time_list <- append(time_list, duration_sec)
###########################################
}


# Get model names from the first column
model_names <- theta_full_db[[1]][1:test_length]
head(model_names)
# Create data frame with model names and theta values
theta_df <- data.frame(Model_Name = model_names, Theta_ATLAS = unlist(theta_atlas_list), Theta_WLE = unlist(theta_full_list), SE = unlist(se_list), Num_Items = unlist(num_items_administered_list), Time_Taken_Sec = unlist(time_list))

# Save results
output_file <- paste0(DIRECTORY, "irt_person_scores_ATLAS_", se_theta_stop, ".csv")#⚠️
write.csv(theta_df, output_file, row.names = FALSE)
print(paste("Saved person scores to", output_file))

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
  
  
  return(item_freq_df)
}

# Analyze item frequency if selected items exist
cat("\n---------- Item Selection Frequency Analysis ----------\n")
if(dir.exists(paste0(DIRECTORY, "selected_items_", se_theta_stop)) && 
   length(list.files(paste0(DIRECTORY, "selected_items_", se_theta_stop), pattern="_items\\.csv$")) > 0) {
  item_freq_results <- analyze_item_frequency()
} else {
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run ATLAS first to generate item selection data.\n")
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
  
  cat("\nItem position analysis saved to ", paste0(DIRECTORY, "item_position_analysis_", se_theta_stop, ".csv"), "\n")
  
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
  cat("No selected item files found in, ",paste0(DIRECTORY, "selected_items_", se_theta_stop), ". Run ATLAS first to generate item selection data.\n")
  position_results <- NULL
}
