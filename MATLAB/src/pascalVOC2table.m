ann_xml_folder='c:\Users\fodrasz\OneDrive\Annotation\IDB_Pylon\pylon_output\Annotations\';
ann_xmls=dir([ann_xml_folder,'*.xml']);
classes={'concretepylon','metalpylon','woodpylon'};
merged_classes={'pylon'}

T=table();
i_obj=0;
jpeg_file=cell(length(ann_xmls),1);
for i=1:length(ann_xmls)
    ann_xml=ann_xmls(i);
    jpeg_file{i}=['c:\Users\fodrasz\OneDrive\Annotation\IDB_Pylon\pylon\',strrep(ann_xml.name,'xml','jpg')];
    % ToDo: check if file exist
    [s] = xml2struct([ann_xml.folder,'\',ann_xml.name]);
    obj={};
    if isfield(s.annotation,'object')
        if length(s.annotation.object)==1
            obj{1}=s.annotation.object;
        else
            obj=s.annotation.object;
        end            
        for j=1:(length(obj))
            name=obj{j}.name;
            disp(name)
            i_obj=i_obj+1;
            xmin=str2double(obj{j}.bndbox.xmin.Text);
            ymin=str2double(obj{j}.bndbox.ymin.Text);
            xmax=str2double(obj{j}.bndbox.xmax.Text);
            ymax=str2double(obj{j}.bndbox.ymax.Text);
            %ind=strcmp(classes,name.Text);
            ind=1;
            %bbox=[xmin,ymin,xmax-xmin,ymax-ymin]
            bboxes=cell(1,length(ind));
            bboxes{1,ind}={[xmin,ymin,xmax-xmin,ymax-ymin]};
            T=[T;[jpeg_file{i},bboxes]];
        end
    end
end

T.Properties.VariableNames = [{'imageFileName'},merged_classes];

