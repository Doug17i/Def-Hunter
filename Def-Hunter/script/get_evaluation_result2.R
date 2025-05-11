library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = '../output/figure/RQ2/'

dir.create(file.path(save.fig.dir), showWarnings = FALSE)

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- do.call(cbind, split(x, x$variable))
  tmp <- tmp[, grep("value", names(tmp))]
  names(tmp) <- gsub(".value", "", names(tmp))
  df <- tmp
  ranking <- NULL
  
  if(reverse == TRUE)
  { 
    ranking <- (max(sk_esd(df)$group)-sk_esd(df)$group) +1 
  }
  else
  { 
    ranking <- sk_esd(df)$group 
  }
  
  x$rank <- paste("Rank",ranking[as.character(x$variable)])
  return(x)
}

# preprocess <- function(x, reverse) {
#
#   colnames(x) <- c("variable", "value")
#   
#
#   split_data <- split(x, x$variable)
#   
#
#   split_data <- lapply(split_data, function(df, var_name) {
#     df$variable <- var_name
#     return(df)
#   }, var_name = names(split_data))
#   
#
#   merged_df <- Reduce(function(df1, df2) {
#     merge(df1, df2, by = "variable", all = TRUE)
#   }, split_data)
#   
#
#   ranking <- NULL
#   if(reverse == TRUE) {
#     ranking <- (max(sk_esd(merged_df)$group) - sk_esd(merged_df)$group) + 1 
#   } else {
#     ranking <- sk_esd(merged_df)$group 
#   }
#   
#
#   rank_mapping <- data.frame(variable = unique(x$variable), rank = ranking)
#   
#
#   x <- merge(x, rank_mapping, by = "variable", all.x = TRUE)
#   
#
#   return(x)
# 
# }
# 



get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'
  
  return(top.k)
}


prediction_dir = '../output/prediction/DeepLineDP/within-release/'
original.prediction_dir = '../output/prediction/Def_Hunter/within-release/'

all_files = list.files(prediction_dir)
original_all_files = list.files(original.prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}


original_df_all <- NULL

for(f in original_all_files)
{
  original_df <- read.csv(paste0(original.prediction_dir, f))
  original_df_all <- rbind(original_df_all, original_df)
}



# ---------------- Code for RQ2 -----------------------#

## prepare data for baseline
line.ground.truth = select(df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth)
line.ground.truth = filter(line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
line.ground.truth = distinct(line.ground.truth)

original.line.ground.truth = select(original_df_all,  project, train, test, filename, file.level.ground.truth, prediction.prob, line.number, line.level.ground.truth)
original.line.ground.truth = filter(original.line.ground.truth, file.level.ground.truth == "True" & prediction.prob >= 0.5)
original.line.ground.truth = distinct(original.line.ground.truth)

get.line.metrics.result = function(baseline.df, cur.df.file)
{
  baseline.df.with.ground.truth = merge(baseline.df, cur.df.file, by=c("filename", "line.number"))
  
  sorted = baseline.df.with.ground.truth %>% group_by(filename) %>% arrange(-line.score, .by_group = TRUE) %>% mutate(order = row_number())
  
  #IFA
  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename)  %>% top_n(1, -order)
  
  ifa.list = IFA$order
  
  total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  
  #Recall20%LOC
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)
  
  recall.list = recall20LOC$recall20LOC
  
  #Effort20%Recall
  effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  
  effort.list = effort20Recall$effort20Recall
  
  result.df = data.frame(ifa.list, recall.list, effort.list)
  
  return(result.df)
}

all_eval_releases = c('activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0', 
                      'camel-2.10.0', 'camel-2.11.0' , 
                      'derby-10.5.1.1' , 'groovy-1_6_BETA_2' , 'hbase-0.95.2', 
                      'hive-0.12.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',  
                      'lucene-3.0.0', 'lucene-3.1', 'wicket-1.5.3')

error.prone.result.dir = '../output/ErrorProne_result/'
ngram.result.dir = '../output/n_gram_result/'
rf.result.dir = '../output/RF-line-level-result/'

n.gram.result.df = NULL
error.prone.result.df = NULL
rf.result.df = NULL 

## get result from baseline
for(rel in all_eval_releases)
{  
  error.prone.result = read.csv(paste0(error.prone.result.dir,rel,'-line-lvl-result.txt'),quote="")
  
  # levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="False"] = 0
  # levels(error.prone.result$EP_prediction_result)[levels(error.prone.result$EP_prediction_result)=="True"] = 1
  
  error.prone.result$EP_prediction_result[error.prone.result$EP_prediction_result=="False"] = 0
  error.prone.result$EP_prediction_result[error.prone.result$EP_prediction_result=="True"] = 1
  # error.prone.result$EP_prediction_result = as.numeric(as.numeric_version(error.prone.result$EP_prediction_result))
  error.prone.result$EP_prediction_result = as.numeric(error.prone.result$EP_prediction_result)
  
  names(error.prone.result) = c("filename","test","line.number","line.score")
  
  n.gram.result = read.csv(paste0(ngram.result.dir,rel,'-line-lvl-result.txt'), quote = "", sep='\t')
  n.gram.result = select(n.gram.result, "file.name", "line.number",  "line.score")
  n.gram.result = distinct(n.gram.result)
  names(n.gram.result) = c("filename", "line.number", "line.score")
  
  rf.result = read.csv(paste0(rf.result.dir,rel,'-line-lvl-result.csv'))
  rf.result = select(rf.result, "filename", "line_number","line.score.pred")
  names(rf.result) = c("filename", "line.number", "line.score")
  
  cur.df.file = filter(line.ground.truth, test==rel)
  cur.df.file = select(cur.df.file, filename, line.number, line.level.ground.truth)
  
#  original.cur.df.file = filter(original.line.ground.truth, test==rel)
#  original.cur.df.file = select(cur.df.file, filename, line.number, original.line.level.ground.truth)
  
  
  n.gram.eval.result = get.line.metrics.result(n.gram.result, cur.df.file)
  
  error.prone.eval.result = get.line.metrics.result(error.prone.result, cur.df.file)
  
  rf.eval.result = get.line.metrics.result(rf.result, cur.df.file)
  
  n.gram.result.df = rbind(n.gram.result.df, n.gram.eval.result)
  error.prone.result.df = rbind(error.prone.result.df, error.prone.eval.result)
  rf.result.df = rbind(rf.result.df, rf.eval.result)
  
  print(paste0('finished ', rel))
  
}
print('finish 1')

#Force attention score of comment line is 0
df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

tmp.top.k = get.top.k.tokens(df_all, 1500)

merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0

print('finish 2')

#Force attention score of comment line is 0 ###
original_df_all[original_df_all$is.comment.line == "True",]$token.attention.score = 0

tmp.top.k = get.top.k.tokens(original_df_all, 1500)

original_merged_df_all = merge(original_df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

original_merged_df_all[is.na(original_merged_df_all$flag),]$token.attention.score = 0

print('finish 2')





## use top-k tokens 
sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())

sorted = sum_line_attn %>% group_by(test, filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())

print('finish 3')

## use top-k tokens ###
original_sum_line_attn = original_merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())

original_sorted = original_sum_line_attn %>% group_by(test, filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(original.order = row_number())

print('finish 3')




## get result from Def-Hunter
# calculate IFA
IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test, filename) %>% top_n(1, -order)

total_true = sorted %>% group_by(test, filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))

# calculate Recall20%LOC
recall20LOC = sorted %>% group_by(test, filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
  summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
  merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

# calculate Effort20%Recall
effort20Recall = sorted %>% merge(total_true) %>% group_by(test, filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
  summarise(effort20Recall = sum(recall <= 0.2)/n())

print('finish 4')

## get result from DeepLineDP
# calculate IFA
original.IFA = original_sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test, filename) %>% top_n(1, -original.order)

original.total_true = original_sorted %>% group_by(test, filename) %>% summarize(original.total_true = sum(line.level.ground.truth == "True"))

# calculate Recall20%LOC
original.recall20LOC = original_sorted %>% group_by(test, filename) %>% mutate(effort = round(original.order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
  summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
  merge(original.total_true) %>% mutate(original.recall20LOC = correct_pred/original.total_true)

# calculate Effort20%Recall
original.effort20Recall = original_sorted %>% merge(original.total_true) %>% group_by(test, filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/original.total_true, digits = 2)) %>%
  summarise(original.effort20Recall = sum(recall <= 0.2)/n())

print('finish 5')

## prepare data for plotting
deeplinedp.ifa = IFA$order
deeplinedp.recall = recall20LOC$recall20LOC
deeplinedp.effort = effort20Recall$effort20Recall

deepline.dp.line.result = data.frame(deeplinedp.ifa, deeplinedp.recall, deeplinedp.effort)

original.ifa = original.IFA$original.order
original.recall = original.recall20LOC$original.recall20LOC
original.effort = original.effort20Recall$original.effort20Recall


length_original_ifa <- length(original.ifa)
print(paste("Length of original.ifa:", length_original_ifa))


length_original_recall <- length(original.recall)
print(paste("Length of original.recall:", length_original_recall))


length_original_effort <- length(original.effort)
print(paste("Length of original.effort:", length_original_effort))


original.line.result = data.frame(original.ifa, original.recall, original.effort)








# 假设 original_df 是原始数据框
n <- nrow(original.line.result)
additional_rows <- 1033 - n

# 创建一个新的空白数据框
blank_df <- data.frame(matrix(NA, ncol = ncol(original.line.result), nrow = additional_rows))
colnames(blank_df) <- colnames(original.line.result)

# 将空白数据框添加到原始数据框
original.line.result <- rbind(original.line.result, blank_df)

print('finish 6')

names(rf.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(n.gram.result.df) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(error.prone.result.df)  = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(deepline.dp.line.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")
names(original.line.result) = c("IFA", "Recall20%LOC", "Effort@20%Recall")


rf.result.df$technique = 'RF'
n.gram.result.df$technique = 'N.gram'
error.prone.result.df$technique = 'ErrorProne'
deepline.dp.line.result$technique = 'Def-Hunter'
original.line.result$technique = 'DeepLineDP'
print('finish 7')

all.line.result = rbind(original.line.result, rf.result.df, n.gram.result.df, error.prone.result.df, deepline.dp.line.result)
# 检查每个数据框的行数
print(paste("Rows in original.line.result:", nrow(original.line.result)))
print(paste("Rows in rf.result.df:", nrow(rf.result.df)))
print(paste("Rows in n.gram.result.df:", nrow(n.gram.result.df)))
print(paste("Rows in error.prone.result.df:", nrow(error.prone.result.df)))
print(paste("Rows in deepline.dp.line.result:", nrow(deepline.dp.line.result)))

# 检查合并后的数据框行数
all.line.result = rbind(original.line.result, rf.result.df, n.gram.result.df, error.prone.result.df, deepline.dp.line.result)
print(paste("Rows in all.line.result after merging:", nrow(all.line.result)))
print('finish 8')

recall.result.df = select(all.line.result, c('technique', 'Recall20%LOC'))
ifa.result.df = select(all.line.result, c('technique', 'IFA'))
effort.result.df = select(all.line.result, c('technique', 'Effort@20%Recall'))
print('finish 9')

recall.result.df = preprocess(recall.result.df, FALSE)
ifa.result.df = preprocess(ifa.result.df, TRUE)
effort.result.df = preprocess(effort.result.df, TRUE)
print('finish 10')

ggplot(recall.result.df, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Recall@Top20%LOC") + xlab("")
ggsave(paste0(save.fig.dir,"file-Recall@Top20LOC.pdf"),width=4,height=2.5)

ggplot(effort.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Effort@Top20%Recall") + xlab("")
ggsave(paste0(save.fig.dir,"file-Effort@Top20Recall.pdf"),width=4,height=2.5)

ggplot(ifa.result.df, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + coord_cartesian(ylim=c(0,175)) + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("IFA") + xlab("")
ggsave(paste0(save.fig.dir, "file-IFA.pdf"),width=4,height=2.5)

