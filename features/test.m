files = dir();
for  i=1:length(files)
    filename = files(i).name;   
    if(~strcmp(filename,'.') && ~strcmp(filename,'..') && strcmp(filename(end-2:end),'mat') && ~strcmp(filename(1:6),'HCTSA-'))
         %type = strsplit(filename,'.')
         %type = char(type(2));  
         filename
         TS_init(filename);
         TS_compute();
         title = filename(1:end-4);
         title = ['HCTSA-',title,'.mat']
        eval(['!rename' , ',HCTSA.mat',[',',title]]); 
    end
end