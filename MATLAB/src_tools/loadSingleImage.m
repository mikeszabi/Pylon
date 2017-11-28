function [I_out]=loadSingleImage(fileName,thumbNail_size)

% output normalized to max intensity: 1

I_out.RGB_RAW=[];
I_out.RGB_SMALL=[];
I_out.fileName='';
I_out.filePath='';
I_out.gtLabel='';
I_out.rating=[];
I_out.dateTick=[];
I_out.info='';

try
    RGB_ORIG=imread(fileName);
    RGB_RAW=im2double(RGB_ORIG);
%     info = getexif(fileName);
    info = imfinfo(fileName);
catch
    disp('not an image file');
    return
end
if length(size(RGB_RAW))>1
    I_out.RGB_RAW=RGB_RAW;
    dummy = regexp(fileName, '\', 'split');
    if ~isempty(thumbNail_size)
        resizeFactor=thumbNail_size/max(size(RGB_RAW)); resizeFactor=min(resizeFactor,1);
        newSize=round(size(RGB_RAW)*resizeFactor);
        try % if openCV is used
            newSize=newSize(2:-1:1); % OpenCV: width x height
            I_out.RGB_SMALL=cv.resize(RGB_RAW,newSize,'Interpolation','Nearest');
        catch
            newSize=newSize(1:2);
            I_out.RGB_SMALL=imresize(RGB_RAW,newSize,'Nearest');
        end
    else
        I_out.RGB_SMALL=[];
    end
    I_out.fileName=char(dummy(length(dummy))); %fileName
    I_out.filePath=strjoin(dummy(1:length(dummy)-1),'\'); %filePath
    I_out.gtLabel=char(dummy(length(dummy)-1)); % container folder name
    I_out.info=info;
    disp([I_out.fileName,'... loaded']);
end

end
    