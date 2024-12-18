function data = getdatafromfile(fp,datanum)
j = sqrt(-1);
data = fscanf(fp,'%f', datanum*2);
data = reshape(data,2,datanum);
data = data(1,:)+j*data(2,:);
end