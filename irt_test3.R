library(bigIRT)


dat<-read.csv("~/llmbenchmark/cleaned_leaderboard_bbh_tracking_shuffled_objects_five_objects_response_matrix.csv")
names(dat)
setnames(dat, "model", "id")
item_cols <- setdiff(names(dat), "id")
item_cols
setDT(dat)
if (!"id" %in% names(dat)) dat[, id := .I]
if ("package:reshape2" %in% search()) detach("package:reshape2", unload = TRUE)

long <-  data.table::melt(
  dat,
  id.vars = "id",
  measure.vars = item_cols,
  variable.name = "Item",
  value.name = "score"
)
long

# 3) Clean types; keep only 0/1 (others -> NA); add Scale
long[, `:=`(
  Item  = as.character(Item),
  score = suppressWarnings(as.integer(score)),
  Scale = 1L
)]
long[!(score %in% c(0L, 1L)), score := NA_integer_]

# Quick check
long[1:6]


set.seed(1)
fit <- bigIRT::fitIRT(
  dat   = long,
  score = "score", id = "id", item = "Item", scale = "Scale",
  pl    = 3,                        # <-- 3PL
  cores = max(1, parallel::detectCores(logical = FALSE) - 1),
  dropPerfectScores = TRUE,         # safety; OK if already dropped
  iter  = 2000                      # increase if needed
)
head(fit)
head(fit$itemPars) 
head(fit$personPars)
## Global & item fit via mirt (diagnostics)
wdat <- dcast(long, id ~ Item, value.var = "score")[, -1]  # back to wide (drop id)
mmod <- mirt(wdat, model = 1, itemtype = "3PL", method = "EM", verbose = FALSE)

M2(mmod, calcNull = TRUE)   # global fit: M2, df, p, RMSEA, SRMSR, CFI/TLI
itemfit(mmod, S_X2 = TRUE)  # item fit: S-X2 p-values and per-item RMSEA