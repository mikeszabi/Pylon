projectDir='C:\Users\fodrasz\Documents\Pylon\MATLAB\';
workDir=[projectDir,'tmp'];
tempdir='D:\tmp\';
%%

% adding source directory
addpath([projectDir,'src']);
% adding tools directory
addpath([projectDir,'src_tools']);
% adding 3rd party directory
addpath_recurse([projectDir,'3rd_party']);

detector_file='C:\Users\fodrasz\Documents\Pylon\MATLAB\pylon_detector.mat';

%%
showFig=true;
cd(workDir)