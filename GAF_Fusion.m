clear all;close all; clear variables; clear global; clc; 
%% IN THIS EXPERIMENT, WE PERFORM ADDITION FUSION 

 %% DEPTH DATA FEATURE EXTRACTION

IMDS1 = imageDatastore('DepthImages_64x64\','IncludeSubfolders',true,....
      'FileExtensions','.png','LabelSource','foldernames');
 
    tbl = countEachLabel(IMDS1);
%Because imds above contains an unequal number of images per category, let's first adjust it,
%so that the number of images in the training set is balanced.

minSetCount = 2500; % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
IMDS1 = splitEachLabel(IMDS1, minSetCount, 'randomize');
  example_image = readimage(IMDS1,1);                      % read one example image
numChannels = size(example_image,3);                    % get color information
numImageCategories = size(categories(IMDS1.Labels),1);
[trainingDS1,validationDS1] = splitEachLabel(IMDS1,0.8,'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS1) ;
load('XONet_DepthImages_64x64.mat');
XONet1=XONet;
XONet1.Layers;

%% Signal images feature extraction


% load the file data for training the CNN
   % use imageDatastore for loading the two image categories 
  IMDS2 = imageDatastore('SignalImages_64x64\','IncludeSubfolders',true,....
      'FileExtensions','.jpg','LabelSource','foldernames');
%     tbl = countEachLabel(IMDS2);
% %Because imds above contains an unequal number of images per category, let's first adjust it,
% %so that the number of images in the training set is balanced.
% 
% minSetCount = 2500; % determine the smallest amount of images in a category
% 
% % Use splitEachLabel method to trim the set.
% IMDS2 = splitEachLabel(IMDS2, minSetCount, 'randomize');
  numImageCategories = size(categories(IMDS2.Labels),1)   % get category labels
[trainingDS2,validationDS2] = splitEachLabel(IMDS2,0.8,'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS2)
load('XONet_SignalImages_64x64.mat');
XONet2=XONet;
XONet2.Layers;

%% conv_1
layer = 'conv_1';
conv1_featuresTrain1 = activations(XONet1,trainingDS1,layer,'OutputAs','rows');
%conv1_featuresTrain11 = activations(XONet1,trainingDS1,layer);
conv1_featuresTest1 = activations(XONet1,validationDS1,layer,'OutputAs','rows');
YTrain1 = trainingDS1.Labels;
YTest1 = validationDS1.Labels;

 conv1_featuresTrain2 = activations(XONet2,trainingDS2,layer,'OutputAs','rows');
 conv1_featuresTest2= activations(XONet2,validationDS2,layer,'OutputAs','rows');
YTrain2 = trainingDS2.Labels;
YTest2 = validationDS2.Labels;

conv1_featuresTrain1 = imresize(conv1_featuresTrain1,size(conv1_featuresTrain2));
conv1_featuresTest1 = imresize(conv1_featuresTest1,size(conv1_featuresTest2));
clear layer
% Fusing training features from conv_1
conv1_XTrain = gaf(conv1_featuresTrain1,conv1_featuresTrain2);
% Fusing test features from conv_
conv1_XTest = gaf(conv1_featuresTest1,conv1_featuresTest2);


%% conv_2
layer = 'conv_2';
conv2_featuresTrain1 = activations(XONet1,trainingDS1,layer,'OutputAs','rows');
conv2_featuresTest1 = activations(XONet1,validationDS1,layer,'OutputAs','rows');

conv2_featuresTrain2 = activations(XONet2,trainingDS2,layer,'OutputAs','rows');
conv2_featuresTest2= activations(XONet2,validationDS2,layer,'OutputAs','rows');

conv2_featuresTrain1 = imresize(conv2_featuresTrain1,size(conv2_featuresTrain2));
conv2_featuresTest1 = imresize(conv2_featuresTest1,size(conv2_featuresTest2));

YTrain2 = trainingDS2.Labels;
YTest2 = validationDS2.Labels;
% Fusing training features from conv_2

conv2_XTrain = gaf(conv2_featuresTrain1,conv2_featuresTrain2);


% Fusing test features from conv_2

conv2_XTest = gaf(conv2_featuresTest1,conv2_featuresTest2);

%% conv_3
layer = 'conv_3';
conv3_featuresTrain1 = activations(XONet1,trainingDS1,layer,'OutputAs','rows');
conv3_featuresTest1 = activations(XONet1,validationDS1,layer,'OutputAs','rows');

conv3_featuresTrain2 = activations(XONet2,trainingDS2,layer,'OutputAs','rows');
conv3_featuresTest2= activations(XONet2,validationDS2,layer,'OutputAs','rows');

conv3_featuresTrain1 = imresize(conv3_featuresTrain1,size(conv3_featuresTrain2));
conv3_featuresTest1 = imresize(conv3_featuresTest1,size(conv3_featuresTest2));
% Fusing training features from conv_3

conv3_XTrain = gaf(conv3_featuresTrain1,conv3_featuresTrain2);


% Fusing test features from conv_3

conv3_XTest = gaf(conv3_featuresTest1,conv3_featuresTest2);


%% fc
layer = 'fc_1';
fc_featuresTrain1 = activations(XONet1,trainingDS1,layer,'OutputAs','rows');
fc_featuresTest1 = activations(XONet1,validationDS1,layer,'OutputAs','rows');

fc_featuresTrain2 = activations(XONet2,trainingDS2,layer,'OutputAs','rows');
fc_featuresTest2= activations(XONet2,validationDS2,layer,'OutputAs','rows');

fc_featuresTrain1 = imresize(fc_featuresTrain1,size(fc_featuresTrain2));
fc_featuresTest1 = imresize(fc_featuresTest1,size(fc_featuresTest2));

% Fusing training features from fc

fc_XTrain = gaf(fc_featuresTrain1,fc_featuresTrain2);

% Fusing test features from fc

fc_XTest = gaf(fc_featuresTest1,fc_featuresTest2);


%% Final Fusion 
conv1_XTrain = imresize(conv1_XTrain,size(fc_XTrain));
conv2_XTrain = imresize(conv2_XTrain,size(fc_XTrain));
conv3_XTrain = imresize(conv3_XTrain,size(fc_XTrain));

 XTrain = [conv1_XTrain conv2_XTrain conv3_XTrain fc_XTrain];
 
 conv1_XTest = imresize(conv1_XTest,size(fc_XTest));
 conv2_XTest = imresize(conv2_XTest,size(fc_XTest));
 conv3_XTest = imresize(conv3_XTest,size(fc_XTest));

 XTest = [conv1_XTest conv2_XTest conv3_XTest fc_XTest];

 %% Fitting Classifier

classifier = fitcecoc(XTrain,YTrain2);
YPred = predict(classifier,XTest);
accuracy = mean(YPred == YTest2);
confMat = confusionmat(YTest2,YPred);
accuracy = mean(YPred == YTest2)
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
plotconfusion(YTest2',YPred','Testing Accuracy');

