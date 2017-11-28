function [I_out_struct]=loadMultipleImage(varargin)

% reads images from a directory recursively
% output normalized images with max intensity: 1
% USES OPENCV !

I_out_struct=[];

inputDir='';
thumbNail_size=600;
imInMemory=false;
jsonFile='';
maxNIm=Inf;

if nargin>0
    inputDir=char(varargin(1));
    if nargin>1
        thumbNail_size=double(varargin{2});
        if nargin>2
            maxNIm=int32(varargin{3});
            if nargin>3
                imInMemory=logical(varargin{4});
                if nargin>4
                    jsonFile=char(varargin(5));
                    if ~isempty(jsonFile)
                        json=NET.addAssembly('e:\MatlabWorks\BPS\JSON\JSON\bin\Release\JSON.dll')
                        disp('loading image data from json');
                        imData=JSON.JSONwrap.GetImages(jsonFile);
                        for i=0:imData.Count-1
                            if imData.Item(i).ImageGroupId.HasValue
                                gtLabel=cast(imData.Item(i).ImageGroupId.Value.ToString,'char');
                            else
                                gtLabel='';
                            end
                            fname=cast(imData.Item(i).ThumbnailFileName.ToString,'char');
                            dateTick=cast(imData.Item(i).DateTaken.Ticks,'double')/1000000;
                            fileNames{i+1}=fname;
                            dateTicks{i+1}=dateTick;
                            gtLabels{i+1}=gtLabel;
                            rating(i+1)=imData.Item(i).Rating;
                        end
                    end
                end
            end
        end
    end
end


%% Initialize output structure
IMCACHESIZE=10000;

clear I_out_struct;
I_out_struct.RGB_RAW=[];
I_out_struct.RGB_SMALL=[];
I_out_struct.fileName='';
I_out_struct.filePath='';
I_out_struct.gtLabel='';
I_out_struct.rating=[];
I_out_struct.dateTick=[];
I_out_struct.info='';

I_out_struct(IMCACHESIZE)=I_out_struct;

if ~imInMemory
    thumbNail_size=[];
end

%% LOAD MODULE

entries = rdir([inputDir,'**\*']); % resursive file collection
iOk=int32(0);
for i=1:length(entries)
    if  ~entries(i).isdir && iOk<maxNIm
        [I_out]=loadSingleImage(entries(i).name,thumbNail_size);
        
        if ~isempty(I_out.RGB_RAW);
            iOk=iOk+1;
            disp(iOk)
            I_out_struct(iOk)=I_out;    
            if ~imInMemory
                I_out_struct(iOk).RGB_RAW=[];
                I_out_struct(iOk).RGB_SMALL=[];
            end
        
            if ~isempty(jsonFile)
                imInd=find(~cellfun(@isempty,strfind(fileNames,I_out.fileName)), 1);

                if ~isempty(imInd)
                        I_out_struct(iOk).gtLabel=gtLabels{imInd};
                        I_out_struct(iOk).rating=rating(imInd)==1; 
                        I_out_struct(iOk).dateTick=dateTicks(imInd);
                end      
            end
        end

    end
end

if iOk>0
    I_out_struct=I_out_struct(1:iOk);
else
    I_out_struct=[];
end