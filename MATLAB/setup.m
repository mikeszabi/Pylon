projectDir='C:\Users\fodrasz\Documents\Pylon\MATLAB\';
workDir=projectDir;
tempdir='D:\tmp\';
mkdir(tempdir);
%%

% adding source directory
addpath([projectDir,'src']);
% adding tools directory
addpath([projectDir,'src_tools']);
% adding 3rd party directory
addpath_recurse([projectDir,'3rd_party']);

cur_detector_file='C:\Users\fodrasz\Documents\Pylon\MATLAB\detectors\pylon_detector.mat';
detector_file='C:\Users\fodrasz\Documents\Pylon\MATLAB\detectors\pylon_detector_20171211.mat';

%%
showFig=true;
cd(workDir)