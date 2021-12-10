set.seed(2167)


library(otrimle)

library(rpart)
library(monmlp)
library(kernlab)
#libraries
library(tseries)
library("tsensembler")

library("forecast")


library(kernlab)


library(factoextra)

library(NbClust)



#mars
library("mda")


### gbm
library(gbm)

###rf
library(randomForest)


library(RSNNS)


#gp gaussian processes
library(kernlab)


#pls
## add ncomp to predict
library(pls)



i=1

repeat{  
  path="/Users/saadalla/Downloads/preds_0812_7pm/"
  
  
  setwd(path)
  
  list.f=list.files(pattern='.csv')
  
  
  
  data.test=read.csv(list.f[[i]])
  
  
  
  
  data.train=read.csv(list.f[[(i+43)]])
  
  
  path="/Users/saadalla/Downloads/test_forecasters_0812_7pm (1)/"
  
  
  setwd(path)
  
  list.f1=list.files(pattern='.csv')
  
  data.m=read.csv(list.f1[[(21*(i-1)+13)]])
  
  list.model=lapply(1:nrow(data.m), function(x) which(data.m[x,]>0))
  
  size.model=sapply(1:nrow(data.m), function(x) length(which(data.m[x,]>0)))
  
  SameElements <- function(a, b) return(identical(sort(a), sort(b)))
  
 
  
  l=sapply(2:nrow(data.m), function(x) SameElements(list.model[[x]],list.model[[(x-1)]]) )
 pos= which(l==F)
  #####
  
  
  train.ts=data.frame(data.train[,1])
  test.ts=data.frame(data.test[,1])
  
  names(train.ts)=names(test.ts)="target"
  
  train.reg=data_lag_prep(train.ts,1,5)
  test.reg=data_lag_prep(test.ts,1,5)
  
  #evaluation
  
  
  n=nrow(data.train)
  val.length=20
  data.val=data.train[((n-val.length+1):n), ]
  data.train.n=data.train[(1:(n-val.length)), ]
  
  H=8
  
  
  predictions.table=data.test[,1:ncol(data.val)]
 
  ens.avg.all.s=function(data.test,predictions.table,list.model,t)
  {  
    predictions.table1=predictions.table
    pred.list=lapply(2:ncol(predictions.table1), function(x)predictions.table1[,x])
    preds=sapply(1:length( pred.list), function(x) pred.list[[x]])
    pred=mean(sapply((list.model[[t]]), function(x) pred.list[[x]][t]))
    
    
    
    return(pred)
  }
  
  pred.ens.all=sapply(1:nrow(data.m),function(t)ens.avg.all(data.test,predictions.table,t))
  
  pred.ens.all.s=sapply(1:nrow(data.m),function(t)ens.avg.all.s(data.test,predictions.table,list.model,t))
  
  rmse(data.test$y[1:119],pred.ens.all)
  
  rmse(data.test$y[1:119],pred.ens.all.s)
  
  
  #stacking framework using Rf
  
  t=1
  
  val.length=20
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  
  
  sel.models=(output.m)
  
  x= as.matrix(sel.models[,2:ncol(sel.models)])
  y= sel.models[,1]
  train.st=sel.models
  train.st1=as.matrix(sel.models)
  formula1=y~.
  
  test.st=predictions.table
  names(test.st)=names(train.st)
  x.test= test.st[,2:ncol(sel.models)]
  y.test= test.st[,1]
  
  library(caret)
  fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 5,                      # number of folds
    savePredictions = 'final'      # saves predictions for optimal tuning parameter
    
  ) 
  
  mlboost=train(formula1,train.st1,method='rf',  trControl = fitControl)
  pred.stacking=predict(mlboost,test.st,ncomp=1)
  rmse(pred.stacking,data.test$y)
  
  
  
  t=1
  
  val.length=10
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  
  
  sel.models=(output.m)[,c(1,list.model[[14]]+1)]
  
  x= as.matrix(sel.models[,2:ncol(sel.models)])
  y= sel.models[,1]
  train.st=sel.models
  train.st1=as.matrix(sel.models)
  formula1=y~.
  
  test.st=predictions.table
  names(test.st)=names(train.st)
  x.test= test.st[,2:ncol(sel.models)]
  y.test= test.st[,1]
  
  library(caret)
  fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 5,                      # number of folds
    savePredictions = 'final'      # saves predictions for optimal tuning parameter
    
  ) 
  
  mlboost=train(formula1,train.st1,method='rf',  trControl = fitControl)
  pred.stacking.s=predict(mlboost,test.st,ncomp=1)
  rmse(pred.stacking.s,data.test$y)
  
  
  # Opera framework 
  library(opera)
  t=1
  
  val.length=20
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,2:ncol(predictions.table)])
  
  Y.new=as.vector(predictions.table[,1])
  
  MLpol <- mixture(Y = Y, experts = X, model = "MLpol", loss.type = "square")
  
  
  pred.mlpol=predict(MLpol,newexperts=X.new,newY=Y.new, type='response')
  

  rmse(pred.mlpol,data.test$y)
  
  
#   l=sapply(2:nrow(data.m), function(x) SameElements(list.model[[x]],list.model[[(x-1)]]) )
#   pos= which(l==F) 
#   
#   library(opera)
#   pred.mlpol.s=NULL
# pos=c(1,pos, nrow(data.test))
# t=1
# for(i in (2:length(pos)))
# { 
#   val.length=20
#   
#   
#   output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
#   output.m=output.m[,c(1,list.model[[pos[(i-1)]]]+1)]
#   X=as.matrix(output.m[,2:ncol(output.m)])
#   
#   Y=as.vector(output.m[,1])
#   
#   X.new=as.matrix(predictions.table[(pos[(i-1)]:pos[(i)]),c(list.model[[pos[(i-1)]]]+1)])
#   
#   Y.new=as.vector(predictions.table[(pos[(i-1)]:pos[(i)]),1])
#   
#   MLpol <- mixture(Y = Y, experts = X, model = "MLpol", loss.type = "square")
#   
#   
#   pred.mlpol.s[(pos[(i-1)]:pos[(i)])]=predict(MLpol,newexperts=X.new,newY=Y.new, type='response')
# 
# }

t=1
val.length=20


output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
output.m=output.m[,c(1,list.model[[1]]+1)]
X=as.matrix(output.m[,2:ncol(output.m)])

Y=as.vector(output.m[,1])

X.new=as.matrix(predictions.table[,c(list.model[[1]]+1)])

Y.new=as.vector(predictions.table[,1])

MLpol <- mixture(Y = Y, experts = X, model = "MLpol", loss.type = "square")


pred.mlpol.s=predict(MLpol,newexperts=X.new,newY=Y.new, type='response')


rmse(pred.mlpol.s,data.test$y)
  
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,2:ncol(predictions.table)])
  
  Y.new=as.vector(predictions.table[,1])
  

  
  fs <- mixture(Y = Y, experts = X, model = "FS", loss.type = "square")
  
  pred.fs=predict(fs,newexperts=X.new,newY=Y.new, type='response')
  
  rmse(pred.fs,data.test$y)
  
  
  
  
  t=1
  
  val.length=20
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  output.m=output.m[,c(1,list.model[[1]]+1)]
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,c(list.model[[1]]+1)])
  
  Y.new=as.vector(predictions.table[,1])
  
  
  
  fs <- mixture(Y = Y, experts = X, model = "FS", loss.type = "square")
  
  pred.fs.s=predict(fs,newexperts=X.new,newY=Y.new, type='response')
  
  rmse(pred.fs.s,data.test$y)
  
  
  
  
  
  
  
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,2:ncol(predictions.table)])
  
  Y.new=as.vector(predictions.table[,1])
  
  ewa <- mixture(Y = Y, experts = X, model = "EWA", loss.type = "square")
  
  pred.ewa=predict(ewa,newexperts=X.new,newY=Y.new, type='response')
  
  rmse(pred.ewa,data.test$y)
  
  
  
  val.length=20
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  output.m=output.m[,c(1,list.model[[1]]+1)]
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,c(list.model[[1]]+1)])
  
  Y.new=as.vector(predictions.table[,1])
  
  ewa <- mixture(Y = Y, experts = X, model = "EWA", loss.type = "square")
  
  pred.ewa.s=predict(ewa,newexperts=X.new,newY=Y.new, type='response')
  
  rmse(pred.ewa.s,data.test$y)
  
  
  
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,2:ncol(predictions.table)])
  
  Y.new=as.vector(predictions.table[,1])
  ogd <- mixture(Y = Y, experts = X, model = "OGD", loss.type = "square")
  
  pred.ogd=pred.ewa=predict(ogd,newexperts=X.new,newY=Y.new, type='response')
  
  
  rmse(pred.ogd,data.test$y)
  
  
  
  
  output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
  output.m=output.m[,c(1,list.model[[1]]+1)]
  X=as.matrix(output.m[,2:ncol(output.m)])
  
  Y=as.vector(output.m[,1])
  
  X.new=as.matrix(predictions.table[,c(list.model[[1]]+1)])
  
  Y.new=as.vector(predictions.table[,1])
  
  
  ogd <- mixture(Y = Y, experts = X, model = "OGD", loss.type = "square")
  
  pred.ogd.s=pred.ewa=predict(ogd,newexperts=X.new,newY=Y.new, type='response')
  
  
  rmse(pred.ogd.s,data.test$y)
  
  
  
  
  pred.ens.all.sw=sapply(1:nrow(data.test),function(t)ens.step.all(data.test,predictions.table,t,H))
  pred.ens.all.sw.s=sapply(1:nrow(data.test),function(t)ens.step.all(data.test,predictions.table[,c(1,list.model[[1]]+1)],t,H))
  
  
  
  rmse(data.test$y,pred.ens.all.sw)
  
  rmse(data.test$y,pred.ens.all.sw.s)
  
  
  

  

  target=data.test$y
  
 # pred.rnn=predictions.table$rnn_d
  
 # pred.cnn=predictions.table$cnn_a
  
 # pred.our=data.test[,((35):ncol(data.test))]
  
  final.preds=data.frame(target,  pred.ens.all.sw,  pred.ens.all.sw.s, pred.ogd,pred.ogd.s,
                         pred.ewa.s,pred.ewa,pred.fs.s,pred.fs,pred.mlpol.s,pred.mlpol)
  
  
  write.csv(final.preds,paste("/Users/saadalla/Desktop/results-ecml22-ag/results",i,".csv",sep=''),row.names = F)

  
  i=i+1
  
  if(i>43){break}
  
}