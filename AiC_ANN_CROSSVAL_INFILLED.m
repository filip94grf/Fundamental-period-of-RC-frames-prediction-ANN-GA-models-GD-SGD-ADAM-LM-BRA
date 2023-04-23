clear; close all; clc;
filename = 'Database source'; %specify path of your database file (BARE FRAMES)
z = xlsread(filename);
%rng(seed) %set integer value to freeze the results
n = randperm(2178);
input = z(n(1:2178),1:5);
target = z(n(1:2178),6);

%Input and output sets
format longg
x = input';
t = target';

%Algorithm
trainFcn = 'trainbr';  %Bayesian regularization

%Neural network architecture
hiddenLayerSize1 = 10;

%rng(seed) %set integer value to freeze the results

%KFOLD CROSSVALIDATION
%Neural network hyperparameters
netcv = fitnet([hiddenLayerSize1], trainFcn);
netcv.performFcn='mse';
netcv.trainParam.epochs = 1000;
netcv.trainParam.goal = 0;
netcv.trainParam.max_fail = 1.00e+010; 
netcv.trainParam.min_grad = 1.00e-010;
netcv.trainParam.mu_inc = 10; 
netcv.trainParam.mu_max = 1.00e+010;
netcv.trainParam.show = 5;
netcv.trainParam.time = inf;

%Activation functions
netcv.layers{1}.transferFcn = 'tansig';
netcv.layers{2}.transferFcn = 'purelin';

k=10;
for i=1:k
    c = cvpartition(length(t),'KFold',k);
    trainingIdx = training(c,i); 
    testIdx = test(c,i); 
    sum(trainingIdx==1);
    sum(testIdx==1);
    XTrain = input(trainingIdx,:)'; 
    YTrain = target(trainingIdx)'; 
    [XTrainN,xscv] = mapminmax(XTrain); 
    [YTrainN,tscv] = mapminmax(YTrain); 
    XTest = input(testIdx,:);   
    YTest = target(testIdx);
    
    %Division strategy
    [traincvInd,valcvInd,testcvInd] = dividerand(length(YTrain),1.00,0,0); 
    netcv.divideParam.trainRatio = 100/100;
    netcv.divideParam.valRatio = 0/100;
    netcv.divideParam.testRatio = 0/100;

    XTestN = mapminmax('apply',XTest',xscv);
    YTestN = mapminmax('apply',YTest',tscv);

    %Training the network
    netcv.trainParam.mu = 0.010; 
    netcv.trainParam.mu_dec = 0.010; 
    [netcv,trcv] = train(netcv,XTrainN,YTrainN);

    anTRAINcv = netcv(XTrainN); 
    aTRAINcv = mapminmax('reverse',anTRAINcv,tscv)'; 
    anTESTcv = netcv(XTestN); 
    aTESTcv = mapminmax('reverse',anTESTcv,tscv);
    OutputCVtrainANN = aTRAINcv(trcv.trainInd)';
    OutputCVtrainVALANN = aTRAINcv(trcv.testInd)';
    ScaledoutputCVtestANN = netcv(XTestN); 
    OriginaloutputCVtestANN = mapminmax('reverse',ScaledoutputCVtestANN,tscv); 

    %Final results for each k-fold
    performanceCV_TRAIN = perform(netcv,aTRAINcv,YTrainN); 
    performanceCV_TEST = perform(netcv,anTESTcv,YTestN) 

    %Plot number of epochs
    numberofepochs=trcv.num_epochs
    bestepoch=trcv.best_epoch;
    
    %Plot number of efficient network parameters
    numberofefficientparameters=trcv.gamk(bestepoch)
    
    %Total number of network parameters
    totalnumberofparameters = size(getwb(netcv))
    
    %Coefficients of correlation R and coefficient of determination
    %R-squared
    R_TEST = corrcoef(aTESTcv,YTest);
    R_TRAIN = corrcoef(aTRAINcv,YTrain);
    R_squared_TEST=R_TEST.*R_TEST;
    R_squared_TRAIN=R_TRAIN.*R_TRAIN;
end