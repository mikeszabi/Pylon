projectDir='C:\Users\fodrasz\Documents\Pylon\MATLAB\';
% adding source directory
addpath([projectDir,'src']);
% adding tools directory
addpath([projectDir,'src_tools']);
% adding 3rd party directory
addpath_recurse([projectDir,'3rd_party']);

picsDir='e:\Pictures\';
workDir=[projectDir,'tmp'];
tempdir='C:\Users\fodrasz\Documents\Pylon\MATLAB\tmp';



%%
showFig=true;
cd(workDir)