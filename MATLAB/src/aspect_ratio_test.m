

act=T.woodpylon;
ar=[];
j=0;

for i = 1:length(act)
   if size(act{i},2)>0
       j=j+1;
       ar(j)=act{i}(4)/act{i}(3);
   end
end