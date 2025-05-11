library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = '../output/figure/RQ5'

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


# ---------------- Code for RQ5 -----------------------#

get.file.level.metrics = function(df.file)
{
  all.gt = df.file$file.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label
  
  confusion.mat = caret::confusionMatrix(factor(all.pred), reference = factor(all.gt))
  # confusion.mat = confusionMatrix(all.pred, reference = all.gt)
  
  bal.acc = confusion.mat$byClass["Balanced Accuracy"]
  AUC = pROC::auc(all.gt, all.prob)
  
  # levels(all.pred)[levels(all.pred)=="False"] = 0
  # levels(all.pred)[levels(all.pred)=="True"] = 1
  # levels(all.gt)[levels(all.gt)=="False"] = 0
  # levels(all.gt)[levels(all.gt)=="True"] = 1
  all.pred[all.pred=="False"] = 0
  all.pred[all.pred=="True"] = 1
  all.gt[all.gt=="False"] = 0
  all.gt[all.gt=="True"] = 1
  
  
  # all.gt = as.numeric_version(all.gt)
  all.gt = as.numeric(all.gt)
  
  # all.pred = as.numeric_version(all.pred)
  all.pred = as.numeric(all.pred)
  
  MCC = mcc(all.gt, all.pred, cutoff = 0.5)
  
  if(is.nan(MCC))
  {
    MCC = 0
  }
  
  eval.result = c(AUC, MCC, bal.acc)
  
  return(eval.result)
}

get.file.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)
  
  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.test.rels = c()
  
  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))
    
    if(method.name == "DP"||method.name == "DefHunter")
    {
      df = as_tibble(df)
      df = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
      
      df = distinct(df)
    }
    
    file.level.result = get.file.level.metrics(df)
    
    AUC = file.level.result[1]
    MCC = file.level.result[2]
    bal.acc = file.level.result[3]
    
    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.test.rels = append(all.test.rels,f)
    
  }
  
  result.df = data.frame(all.auc,all.mcc,all.bal.acc)
  
  
  all.test.rels = str_replace(all.test.rels, ".csv", "")
  
  result.df$release = all.test.rels
  result.df$technique = method.name
  
  return(result.df)
}

bi.lstm.prediction.dir = "../output/prediction/Bi-LSTM/"
cnn.prediction.dir = "../output/prediction/CNN/"

dbn.prediction.dir = "../output/prediction/DBN/"
bow.prediction.dir = "../output/prediction/BoW/"
# lr.prediction.dir = "../output/prediction/LR/"


original.result = get.file.level.eval.result(original.prediction_dir, "DP")
bi.lstm.result = get.file.level.eval.result(bi.lstm.prediction.dir, "Bi.LSTM")
cnn.result = get.file.level.eval.result(cnn.prediction.dir, "CNN")
dbn.result = get.file.level.eval.result(dbn.prediction.dir, "DBN")
# lr.result = get.file.level.eval.result(lr.prediction.dir, "LR")
bow.result = get.file.level.eval.result(bow.prediction.dir, "BoW")
deepline.dp.result = get.file.level.eval.result(prediction_dir, "DefHunter")

# all.result = rbind(bi.lstm.result, cnn.result, dbn.result, lr.result, deepline.dp.result)
all.result = rbind(original.result, bi.lstm.result, cnn.result, dbn.result, bow.result, deepline.dp.result)
names(all.result) = c("AUC","MCC","Balance.Accuracy","Release", "Technique")

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)
auc.result[auc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)
mcc.result[mcc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)
bal.acc.result[bal.acc.result$variable=="Bi.LSTM", "variable"] = "Bi-LSTM"

ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("AUC") + xlab("")
ggsave(paste0(save.fig.dir,"file-AUC.pdf"),width=4,height=2.5)

ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("Balance Accuracy") + xlab("")
ggsave(paste0(save.fig.dir,"file-Balance_Accuracy.pdf"),width=4,height=2.5)

ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + ylab("MCC") + xlab("")
ggsave(paste0(save.fig.dir, "file-MCC.pdf"),width=4,height=2.5)


