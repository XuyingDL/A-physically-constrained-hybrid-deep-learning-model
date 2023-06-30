clc;
clear;
close all;

I = double(imread('F:\NWHubei\composit_rank\composite.tif'));
[m, n, z] = size(I);

TR = double(imread('F:\NWHubei\data\TR.tif'));
VAL = double(imread('F:\NWHubei\data\VAL.tif'));
TE = double(imread('F:\NWHubei\data\TR.tif'));
W = double(imread('F:\NWHubei\data\fault_rou.tif'));

I2d = hyperConvert2d(I);

TR2d = hyperConvert2d(TR);
VAL2d = hyperConvert2d(VAL);
TE2d = hyperConvert2d(TE);
W2d = hyperConvert2d(W);

TR_sample = I2d(:,TR2d<10);
VAL_sample = I2d(:,VAL2d<10);
TE_sample = I2d(:,TE2d<11);

W_TR = W2d(:,TR2d<10);
W_VAL = W2d(:,VAL2d<10);  
W_TE = W2d(:,TE2d<11);

TR_temp = TR2d(:,TR2d<10);
VAL_temp = VAL2d(:,VAL2d<10);


Train_X = TR_sample';
Val_X = VAL_sample';
Test_X = TE_sample';

TrLabel = TR_temp';
ValLabel = VAL_temp';


%% Please replace the following route with your own one
save('F:\NWHubei\code\1DCNN\Train_X.mat','Train_X');
save('F:\NWHubei\code\1DCNN\Val_X.mat','Val_X');
save('F:\NWHubei\code\1DCNN\Test_X.mat','Test_X');
save('F:\NWHubei\code\1DCNN\TrLabel.mat','TrLabel');
save('F:\NWHubei\code\1DCNN\ValLabel.mat','ValLabel');
save('F:\NWHubei\code\W\W_TR.mat','W_TR');
save('F:\NWHubei\code\W\W_VAL.mat','W_VAL');
