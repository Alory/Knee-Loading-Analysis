files = dir();
features = [];
ts = [];
var = 'TS_DataMat';

for  i=1:length(files)
    filename = files(i).name;   
    if(~strcmp(filename,'.') && ~strcmp(filename,'..') && strcmp(filename(end-2:end),'mat') && strcmp(filename(1:6),'HCTSA-'))
        filename 
        txtname = filename(1:end-4)
         
        load(filename);
         
         dlmwrite(['feature-',txtname,'.txt'], TS_DataMat, 'precision', '%6f', 'delimiter', '\t');
         %dlmwrite(['rawdata-',txtname,'.txt'], TimeSeries, 'precision', '%6f', 'delimiter', '\t');
         
         %TS_init(filename);
         %TS_compute();
         %title = filename(1:end-4);
         %title = ['HCTSA-',title,'.mat']
         %eval(['!rename' , ',HCTSA.mat',[',',title]]); 
    end
end