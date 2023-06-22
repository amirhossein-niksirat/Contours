clc;clear;close all
%% Input and target Data
x1=0:0.1:2*pi; %theta1
x2=-pi:0.1:pi; % %AOA
x3=-pi:0.1:pi;
%x3=1:63; % beta
x=[x1;x2;x3];
CL=sin(x1).*cos(x2)+x3/100; % Targets

y=CL;
%y=sin(x1)+cos(x2);

%% Network Structure (Hidden Layers)
HLS=[3]; % HLS: Hidden Layers Size is a Vector! --> numel(HLS) : Number of Layers and Number of Neurons is Value of arguments
net=fitnet(HLS);
net.layers{1}.transferFcn='tansig';
net.layers{2}.transferFcn='logsig';
%% Net parameters
net.divideParam.trainRatio=70/100;
net.divideParam.valRatio=15/100;
net.divideParam.testRatio=15/100;
% net.performFcn='mse';
% net.divideFcn='divideind';
% net.divideParam.trainInd=[1:20,30:numel(x)];
% net.divideParam.valInd=[21:25];
% net.divideParam.testInd=[26:29];
%% Train the network
[net,tr]=train(net,x,y);
view(net)
%%
yhat=net(x);
e=y-yhat;
mse=mean(e.^2);
rmse=sqrt(mse);
%% Plots
 plotregression(y,yhat)
 plotfit(net,x,y)
 ploterrhist(e)
%%
figure
plot(x(tr.trainInd),y(tr.trainInd),'ob','markerfacecolor','b')
hold on
plot(x(tr.valInd),y(tr.valInd),'sg','markerfacecolor','g')
plot(x(tr.testInd),y(tr.testInd),'*r','markerfacecolor','r')







