%only bpsk 2ms(1ms相干)+4次非相干
clear all;
simulatecount=100;
Freq_sample = (16.368e6); %采样频率
Freq_IF = 1e6; % 低中频无多普勒频偏载波频率
Freq_code =  2.046e6;% 扩频码频率
codelength = 2046;% 扩频码码长
Frf = 1522224000;% 射频频率
%%捕获参数设定
LocalDop = 29900;     %设定本地复现载波多普勒主频点
AcqSatNUM=1;          %PRN号
freq_dot(AcqSatNUM)=0;%先验信息多普勒一阶变化率
totalnum =9;          %总共使用的数据，60表示60个扩频码周期的数据
seg=22;               %1ms的分段数
fftNUM=128;           %fft点数
%LocalCorreArray = [0:0.5:2045.5];%[-4:0.2:4];  %本地支路数L 本地码搜索码片范围和码片间距(1/8码片精度)  用来遍历码相位
LocalCorreArray=[0:0.5:100];
LengthCorreArray = length(LocalCorreArray);
carrierfretocodefre=Frf/Freq_code;
codeperiod = 0.001; %扩频码周期，单位秒
filename = "LEO_SIGNAL_TEST.txt"; %读取文件
fp = fopen(filename, 'r'); % 打开文件以读取数据
cn0 = 40;
sigma=1;
A = sigma* sqrt(2*10^(cn0/10)/Freq_sample);

%%%%%%%% 调用码发生器函数生成卫星的伪码
[codetemphigh,codetemplow]=CAgenerate(AcqSatNUM);
codelow(1,:)=codetemplow;
codezeros=zeros(1,2046);
codetwo=[codelow codezeros];
j = sqrt(-1);

for i=1:1:simulatecount
    i
    rng(i);
    coh =[];
    nocoh =zeros(LengthCorreArray,fftNUM);%存放非相干累加结果
    cohfft=zeros(LengthCorreArray,fftNUM);
    noncoh = zeros(LengthCorreArray,fftNUM);
    
    fseek(fp, 0, 'bof')  ;%将文件指针移动到文件开头
    datanum_onecodeperiod = round(Freq_sample * codeperiod);  %一个扩频码周期的采样点=扩频码周期×采样率
    carrier_dco_delta = (Freq_IF + LocalDop+freq_dot(AcqSatNUM)*0.001/2)/Freq_sample;  %初始化载波  载波步进，每个采样点载波增量
    carrier_dco_phase_twocodeperiod = carrier_dco_delta*(1:datanum_onecodeperiod*2);  %每个载波的载波相位两个周期
    carrier_dco_phase_zengliang = carrier_dco_delta * datanum_onecodeperiod;   %每个扩频码周期的载波增量
    code_dco_delta = (Freq_code+LocalDop/carrierfretocodefre)/Freq_sample;  %初始化扩频码  码相位步进
    code_dco_phase_twocodeperiod = code_dco_delta*(1:datanum_onecodeperiod*2);   %每个采样点码的相位
    code_dco_phase_zengliang = code_dco_delta * datanum_onecodeperiod;
    code_dco_phase_EPL=repmat(code_dco_phase_twocodeperiod,LengthCorreArray,1);%支路码相位矩阵
    
    for index=1:1:LengthCorreArray
        tmptmp2(index,:) = (code_dco_phase_EPL(index,:)) + LocalCorreArray(index);% 码片各个支路的码相位
        tmptmp = floor(mod(tmptmp2+codelength*2,codelength*2))+1;    %本地扩频码组索引
    end
    for ms = 1:1:totalnum/2
        
        code_dco_phase_EPL=repmat(code_dco_phase_twocodeperiod,LengthCorreArray,1);%支路码相位矩阵
        for index=1:1:LengthCorreArray
            tmptmp2(index,:) = (code_dco_phase_EPL(index,:)) + LocalCorreArray(index);% 码片各个支路的码相位
            tmptmp = floor(mod(tmptmp2+codelength*2,codelength*2))+1;    %本地扩频码组索引
        end
        inputdata_twocodeperiod = A*getdatafromfile(fp,datanum_onecodeperiod*2)+randn(1,datanum_onecodeperiod*2)+sqrt(-1)*randn(1,datanum_onecodeperiod*2);
        %2ms点对点相关
        onemssum=inputdata_twocodeperiod.*exp(-j*2*pi*carrier_dco_phase_twocodeperiod).*codetwo(tmptmp);
        %2ms的采样点分组累加
        %splitMatrixAndSum 第二个变量为该累加周期内的分段数。
        rowSums = splitMatrixAndSum(onemssum, seg*2);
        for index = 1:1:LengthCorreArray
            coh(index,:) = fft(rowSums(index,:), fftNUM); % 每行进行K点FFT
        end
        coh(:,:)=abs(coh(:,:));
        nocoh=nocoh+coh;
        onemssum=0;
        code_dco_phase_twocodeperiod=code_dco_phase_twocodeperiod+code_dco_phase_zengliang*2;
        carrier_dco_phase_twocodeperiod=carrier_dco_phase_twocodeperiod+carrier_dco_phase_zengliang*2;
    end
    %统计峰值
%     figure
%     surf(nocoh)
    fidr = fopen('re.txt','a');
    %找最大峰值处的码相位及多普勒频率
    [maxValue, linearIndex] = max(nocoh(:));  % 返回最大值和线性索引
    [row, col] = ind2sub(size(nocoh), linearIndex);  % 将线性索引转换为行和列索引
    f1=1000/(fftNUM*1/seg);%分辨率
    maxdop=LocalDop+(col-1)*f1;
   fprintf(fidr,'bpsk only cn0 %d   max = %f  fdop= %f Hz  codei = %05d  \n',...
        cn0,maxValue ,maxdop,row);
    fclose(fidr);
end
fclose(fp);








