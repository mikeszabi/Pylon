function [I_out]=load_one_image(fileName,thumbNail_size)

% output normalized to max intensity: 1

I_out.RGB_RAW=[];
I_out.RGB_SMALL=[];
I_out.fileName='';
I_out.filePath='';
I_out.info='';

try
    info = imfinfo(fileName);
    RGB_ORIG=imread(fileName);
    if info.Orientation==6
        RGB_ORIG=imrotate(RGB_ORIG,270);
        % http://sylvana.net/jpegcrop/exif_orientation.html
    end
    RGB_RAW=im2double(RGB_ORIG);
%     info = getexif(fileName);
% UTM
    %dd.ff = dd + mm/60 + ss/3600
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
        newSize=newSize(2:-1:1); % OpenCV: width x height
        I_out.RGB_SMALL=cv.resize(RGB_RAW,newSize,'Interpolation','Nearest');
    else
        I_out.RGB_SMALL=[];
    end
    I_out.fileName=char(dummy(length(dummy))); %fileName
    I_out.filePath=strjoin(dummy(1:length(dummy)-1),'\'); %filePath
    I_out.info=info;
    disp([I_out.fileName,'... loaded']);
end

end