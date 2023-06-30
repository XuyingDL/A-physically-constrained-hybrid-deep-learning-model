clc;
clear;
close all;

I = double(imread('F:\NWHubei\composit_rank\composite.tif'));
[m, n, z] = size(I);

TR = double(imread('F:\NWHubei\data\TR.tif'));
VAL = double(imread('F:\NWHubei\data\VAL.tif'));
TE = double(imread('F:\NWHubei\data\TR.tif'));

I2d = hyperConvert2d(I);

TR2d = hyperConvert2d(TR);
VAL2d = hyperConvert2d(VAL);
TE2d = hyperConvert2d(TE);

TR_sample = I2d(:,TR2d<10);
VAL_sample = I2d(:,VAL2d<10);
TE_sample = I2d(:,TE2d<11);

TR_temp = TR2d(:,TR2d<10);
VAL_temp = VAL2d(:,VAL2d<10);


K = 8;
si = 1;
Train_W = creatLap(TR_sample, K, si);
Train_D = (sum(Train_W, 2)).^(-1/2);
Train_D = diag(Train_D);
L_temp = Train_W * Train_D;
Train_L = Train_D * L_temp;
Train_L = Train_L + eye(size(Train_L));

VAL_W = creatLap(VAL_sample, K, si); 
VAL_D = (sum(VAL_W, 2)).^(-1/2);
VAL_D = diag(VAL_D);
L_temp = VAL_W * VAL_D;
VAL_L = VAL_D * L_temp;
VAL_L = VAL_L + eye(size(VAL_L));

Test_W = creatLap(TE_sample, K, si); 
Test_D = (sum(Test_W, 2)).^(-1/2);
Test_D = diag(Test_D);
L_temp = Test_W * Test_D;
Test_L = Test_D * L_temp;
Test_L = Test_L + eye(size(Test_L));

Train_X = TR_sample';
Val_X = VAL_sample';
Test_X = TE_sample';

TrLabel = TR_temp';
ValLabel = VAL_temp';


%% Please replace the following route with your own one
save('F:\NWHubei\code\miniGCN10\Train_X.mat','Train_X');
save('F:\NWHubei\code\miniGCN10\Val_X.mat','Val_X');
save('F:\NWHubei\code\miniGCN10\Test_X.mat','Test_X');
save('F:\NWHubei\code\miniGCN10\TrLabel.mat','TrLabel');
save('F:\NWHubei\code\miniGCN10\ValLabel.mat','ValLabel');
save('F:\NWHubei\code\miniGCN10\Train_L.mat','Train_L');
save('F:\NWHubei\code\miniGCN10\Val_L.mat','VAL_L');
save('F:\NWHubei\code\miniGCN10\Test_L.mat','Test_L');