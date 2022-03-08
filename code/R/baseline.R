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
path="/Users/saadalla/Downloads/preds_0812_7pm//"


setwd(path)

list.f=list.files(pattern='.csv')



data.test=read.csv(list.f[[i]])




data.train=read.csv(list.f[[(i+43)]])



#####


train.ts=data.frame(data.train[,1])
test.ts=data.frame(data.test[,1])

names(train.ts)=names(test.ts)="target"

train.reg=data_lag_prep(train.ts,1,5)
test.reg=data_lag_prep(test.ts,1,5)

#evaluation

pred.a=calculate_arima_prediction(train.ts, test.ts,nrow(test.ts),124)
rmse(pred.a,data.test$y)
plot(test.ts$target,type='l')
lines(pred.a,col='red')
pred.es=sapply(1:nrow(test.ts), function(x)calculate_ets_prediction(train.ts, test.ts,nrow(test.ts),(x)))
rmse(pred.es,data.test$y)
plot(test.ts$target,type='l')
lines(pred.es,col='red')
n=nrow(data.train)
val.length=20
data.val=data.train[((n-val.length+1):n), ]
data.train.n=data.train[(1:(n-val.length)), ]
formula=target~.
models=train.models(train.reg, formula,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))

predictions.table1=predict.models(models,test.reg,formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))


er.ref=sapply(1:ncol(predictions.table1), function(x) rmse(predictions.table1[,1],predictions.table1[,x]))

names(er.ref)=names(predictions.table1)
tp1=4
lim=0.1
val.length=10

updated_selection=topk.model.sel(models,data.train, data.test,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"),lim,tp1)

updated_selection1=updated_selection$models.sel

alarm=updated_selection$alarm
H=8


predictions.table=data.test[,1:ncol(data.val)]
sapply(1:ncol(predictions.table), function(x) rmse(predictions.table[,1],predictions.table[,x]))

pred.top=ens.top.pred(data.test,predictions.table,updated_selection1,H)

rmse(pred.top,data.test$y)

pred.ens.all.sw=sapply(1:nrow(data.test),function(t)ens.step.all(data.test,predictions.table,t,H))

pred.ens.all=sapply(1:nrow(data.test),function(t)ens.avg.all(data.test,predictions.table,t))



rmse(data.test$y,pred.top)
rmse(data.test$y,pred.ens.all)

rmse(data.test$y,pred.ens.all.sw)


val.length1=10
cluster.res=update.cluster(models,data.train, data.test,val.length1, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"),updated_selection,tp1)

pos.cl=lapply(1:length(cluster.res) ,function(x) if(length(cluster.res[[x]]$list.position)>=4){
  cluster.res[[x]]$list.position[1:4]
}else{c(cluster.res[[x]]$list.position[1],rep(cluster.res[[x]]$list.position[2],3))})


pred.cl=ens.top.pred(data.test,predictions.table,pos.cl,H)

st=Sys.time()
val.length1=10
tp.cl.all=compute.cluster.tp.all(models,data.train,data.test,val.length1,H, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"),updated_selection)
ed=Sys.time()

pred.tp.cl.all=tp.cl.all[[1]]

pos.models=tp.cl.all[[2]]

rmse(data.test$y,pred.cl)


rmse(data.test$y,pred.tp.cl.all)




val.length1=10

pred.cl.or=compute.cluster.or.all(models,data.train, data.test,val.length1,H, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))





val.length1=10
pred.tw.tp=compute.cluster.tw.tp.all(models,data.train, data.test,t,val.length1, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"),updated_selection)




pred.cl.st=pred.tp.cl.all





#stacking framework using Rf

# t=1
# 
# val.length=20
# output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
# 
# 
# sel.models=(output.m)
# 
# x= as.matrix(sel.models[,2:ncol(sel.models)])
# y= sel.models[,1]
# train.st=sel.models
# train.st1=as.matrix(sel.models)
# formula1=y~.
# 
# test.st=predictions.table
# names(test.st)=names(train.st)
# x.test= test.st[,2:ncol(sel.models)]
# y.test= test.st[,1]
# 
# library(caret)
# fitControl <- trainControl(
#   method = 'cv',                   # k-fold cross validation
#   number = 5,                      # number of folds
#   savePredictions = 'final'      # saves predictions for optimal tuning parameter
#   
# ) 
# 
# mlboost=train(formula1,train.st1,method='rf',  trControl = fitControl)
# pred.stacking=predict(mlboost,test.st,ncomp=1)
# rmse(pred.stacking,data.test$y)
# 
# 
# # Opera framework 
# library(opera)
# t=1
# 
# val.length=20
# 
# 
# output.m=drift.topk.models.input(models,data.train, data.test,t,val.length, formula, per.arima,ker=c("rbfdot" ,"polydot" ,"vanilladot", "laplacedot"))
# 
# X=as.matrix(output.m[,2:ncol(output.m)])
# 
# Y=as.vector(output.m[,1])
# 
# X.new=as.matrix(predictions.table[,2:ncol(predictions.table)])
# 
# Y.new=as.vector(predictions.table[,1])
# 
# MLpol <- mixture(Y = Y, experts = X, model = "MLpol", loss.type = "square")
# 
# 
# pred.mlpol=predict(MLpol,newexperts=X.new,newY=Y.new, type='response')
# 
# rmse(pred.mlpol,data.test$y)
# fs <- mixture(Y = Y, experts = X, model = "FS", loss.type = "square")
# 
# pred.fs=predict(fs,newexperts=X.new,newY=Y.new, type='response')
# 
# rmse(pred.fs,data.test$y)
# 
# rmse(pred.tp.cl.all,data.test$y)
# 
# 
# ewa <- mixture(Y = Y, experts = X, model = "EWA", loss.type = "square")
# 
# pred.ewa=predict(ewa,newexperts=X.new,newY=Y.new, type='response')
# 
# rmse(pred.ewa,data.test$y)
# 
# ogd <- mixture(Y = Y, experts = X, model = "OGD", loss.type = "square")
# 
# pred.ogd=pred.ewa=predict(ogd,newexperts=X.new,newY=Y.new, type='response')
# 
# 
# rmse(pred.ogd,data.test$y)
# 
# 




target=data.test$y

pred.rnn=predictions.table$as01_e

pred.cnn=predictions.table$cnn_a

pred.our=data.test[,((35):ncol(data.test))]
# path="/Users/saadalla/Downloads/test_forecasters_0412_1pm/"
# 
# 
# setwd(path)
# 
# list.f1=list.files(pattern='.csv')
# 
# data.m=read.csv(list.f1[[(11*(i-1)+2)]])
# 
# list.model=lapply(1:nrow(data.m), function(x) which(data.m[x,]>0))
# 
# size.model=sapply(1:nrow(data.m), function(x) length(which(data.m[x,]>0)))
# 
# pred.ops1=sapply(1:nrow(data.test),function(t)ens.step.all.s(data.test,predictions.table,t,H))

#rmse(data.test$y,pred.ops1)

final.preds=data.frame(target, pred.rnn, pred.cnn,pred.ens.all,pred.a,pred.es
                       ,pred.cl.or,pred.top,pred.tw.tp, pred.tp.cl.all,pred.our)


write.csv(final.preds,paste("/Users/saadalla/Desktop/results-ecml22/results",i,".csv",sep=''),row.names = F)
# rmse.table=sapply(1:ncol(final.preds), function(x) rmse(final.preds[,1],final.preds[,x] ))
# names(rmse.table)=names(final.preds)
# sapply(2:34, function(x) rmse(data.test$y,data.test[,x] ))
# names(data.test[,2:34])
# names(rmse.table)=names(final.preds)
# 
# 
# mse.table=sapply(1:ncol(final.preds), function(x) mse(final.preds[,1],final.preds[,x] ))
# names(mse.table)=names(final.preds)
# 
# 
# wilcoxon.table=sapply(1:ncol(final.preds), function(x) Wilcoxon.test(final.preds[,1],final.preds[,20],final.preds[,x],124))
# 
# 
# tests=round(t(wilcoxon.table),digits = 4)
# rownames(tests)=names(final.preds)
# 
# 
# 
# alpha1=sapply(1:ncol(final.preds), function(x) if(tests[x,1]<0.5){1}else{0})
# names(alpha1)=names(wilcoxon.table)
# 
# alpha2=sapply(1:ncol(final.preds), function(x) if(tests[x,2]<0.5){1}else{0})
# names(alpha2)=names(wilcoxon.table)
# 
# 
# 
# beta1=sapply(1:ncol(final.preds), function(x) if(tests[x,1]<0.05){1}else{0})
# names(beta1)=names(wilcoxon.table)
# 
# beta2=sapply(1:ncol(final.preds), function(x) if(tests[x,2]<0.05){1}else{0})
# names(beta2)=names(wilcoxon.table)

i=i+1

if(i>43){break}

}