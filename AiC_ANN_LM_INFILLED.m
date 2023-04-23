clear; close all; clc;
filename = 'Database source'; %specify path of your database file (INFILLED FRAMES)
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
trainFcn = 'trainlm';  %Levenberg-Marquardt

%Neural network architecture and hyperparameters
hiddenLayerSize1 = 10;
net = fitnet([hiddenLayerSize1], trainFcn);
net.performFcn='mse';
net.trainParam.epochs = 1000;
net.trainParam.goal = 0;
net.trainParam.max_fail = 10;
net.trainParam.mem_reduc = 1;
net.trainParam.min_grad = 1.00e-010;
net.trainParam.mu = 0.0010; 
net.trainParam.mu_dec = 0.0010; 
net.trainParam.mu_inc = 10; 
net.trainParam.mu_max = 1.00e+010;
net.trainParam.show = 5;
net.trainParam.time = inf;

%Activation functions
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

%Division strategy
[trainInd,valInd,testInd] = dividerand(2178,0.70,0.15,0.15);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

x_train = x(:,trainInd);
t_train = t(trainInd);

x_test = x(:,testInd);
t_test = t(testInd);

%Train the Network
[xn,xs] = mapminmax(x_train);
[tn,ts] = mapminmax(t_train);
[net,tr] = train(net,xn,tn);

%Test the Network
x_testn=mapminmax('apply',x_test,xs);
t_testn=mapminmax('apply',t_test,ts);

xfulln=mapminmax('apply',x,ts); 
y = net(xfulln); 
yrev=mapminmax('reverse',y,ts);


an = sim(net,xn); 
a = mapminmax('reverse',an,ts); 

an_test=sim(net,x_testn);
an_t=mapminmax('reverse',an_test,ts);

%Error values
%e = gsubtract(a,y);
performanceTRAINSET = perform(net,a,t_train);
performanceALLDATA = perform(net,yrev,t);
performanceTESTSET=perform(net,an_t,t_test);
% View the Network
%view(net)
% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotfit(net,x,t)

OutputtrainDATASET = t_train'; 
OutputtrainANN = a'; 
InputtrainDATASETandANN = x_train';

%OutputvalidationDATASET = t(tr.valInd)'; 
%OutputvalidationANN = a(tr.valInd)'; 
%InputvalidationDATASETandANN = x(tr.valInd)';

OutputtestDATASET = t_test'; 
OutputtestANN = an_t'; 
InputtestDATASETandANN = x_test';

OutputALLDATASET = t;
OutputALLDATASETSCALED = y;
InputALLDATASETandANN = x;
InputALLDATASETandANNSCALED = xfulln;

%Performance functions
performanceALLDATA = perform(net,an,tn)
performanceTRAININGSET = perform(net,OutputtrainANN,OutputtrainDATASET)
%performanceVALIDATIONSET = perform(net,OutputvalidationANN,OutputvalidationDATASET); 
performanceTESTSET = perform(net,OutputtestANN,OutputtestDATASET)

performanceALLSETMAE = mae(net,an,tn)
performanceTRAININGSETMAE = mae(net,OutputtrainANN,OutputtrainDATASET)
performanceTESTSETMAE = mae(net,OutputtestANN,OutputtestDATASET)

performanceALLSETRMSE = sqrt(performanceALLDATA)
performanceTRAININGSETRMSE = sqrt(performanceTRAININGSET)
performanceTESTSETRMSE = sqrt(performanceTESTSET)

%R-SQUARED graphs
%figure, plotregression(a,t)
figure, plotregression(OutputtestANN,OutputtestDATASET)

w1 = net.IW{1}; %the input-to-hidden layer weights
w2 = net.LW{2}; %the hidden-to-output layer weights
b1 = net.b{1}; %the input-to-hidden layer bias
b2 = net.b{2}; %the hidden-to-output layer bias