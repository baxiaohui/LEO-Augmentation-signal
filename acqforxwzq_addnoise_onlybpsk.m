%only bpsk 2ms(1ms���)+4�η����
clear all;
simulatecount=100;
Freq_sample = (16.368e6); %����Ƶ��
Freq_IF = 1e6; % ����Ƶ�޶�����Ƶƫ�ز�Ƶ��
Freq_code =  2.046e6;% ��Ƶ��Ƶ��
codelength = 2046;% ��Ƶ���볤
Frf = 1522224000;% ��ƵƵ��
%%��������趨
LocalDop = 29900;     %�趨���ظ����ز���������Ƶ��
AcqSatNUM=1;          %PRN��
freq_dot(AcqSatNUM)=0;%������Ϣ������һ�ױ仯��
totalnum =9;          %�ܹ�ʹ�õ����ݣ�60��ʾ60����Ƶ�����ڵ�����
seg=22;               %1ms�ķֶ���
fftNUM=128;           %fft����
%LocalCorreArray = [0:0.5:2045.5];%[-4:0.2:4];  %����֧·��L ������������Ƭ��Χ����Ƭ���(1/8��Ƭ����)  ������������λ
LocalCorreArray=[0:0.5:100];
LengthCorreArray = length(LocalCorreArray);
carrierfretocodefre=Frf/Freq_code;
codeperiod = 0.001; %��Ƶ�����ڣ���λ��
filename = "LEO_SIGNAL_TEST.txt"; %��ȡ�ļ�
fp = fopen(filename, 'r'); % ���ļ��Զ�ȡ����
cn0 = 40;
sigma=1;
A = sigma* sqrt(2*10^(cn0/10)/Freq_sample);

%%%%%%%% �����뷢���������������ǵ�α��
[codetemphigh,codetemplow]=CAgenerate(AcqSatNUM);
codelow(1,:)=codetemplow;
codezeros=zeros(1,2046);
codetwo=[codelow codezeros];
j = sqrt(-1);

for i=1:1:simulatecount
    i
    rng(i);
    coh =[];
    nocoh =zeros(LengthCorreArray,fftNUM);%��ŷ�����ۼӽ��
    cohfft=zeros(LengthCorreArray,fftNUM);
    noncoh = zeros(LengthCorreArray,fftNUM);
    
    fseek(fp, 0, 'bof')  ;%���ļ�ָ���ƶ����ļ���ͷ
    datanum_onecodeperiod = round(Freq_sample * codeperiod);  %һ����Ƶ�����ڵĲ�����=��Ƶ�����ڡ�������
    carrier_dco_delta = (Freq_IF + LocalDop+freq_dot(AcqSatNUM)*0.001/2)/Freq_sample;  %��ʼ���ز�  �ز�������ÿ���������ز�����
    carrier_dco_phase_twocodeperiod = carrier_dco_delta*(1:datanum_onecodeperiod*2);  %ÿ���ز����ز���λ��������
    carrier_dco_phase_zengliang = carrier_dco_delta * datanum_onecodeperiod;   %ÿ����Ƶ�����ڵ��ز�����
    code_dco_delta = (Freq_code+LocalDop/carrierfretocodefre)/Freq_sample;  %��ʼ����Ƶ��  ����λ����
    code_dco_phase_twocodeperiod = code_dco_delta*(1:datanum_onecodeperiod*2);   %ÿ�������������λ
    code_dco_phase_zengliang = code_dco_delta * datanum_onecodeperiod;
    code_dco_phase_EPL=repmat(code_dco_phase_twocodeperiod,LengthCorreArray,1);%֧·����λ����
    
    for index=1:1:LengthCorreArray
        tmptmp2(index,:) = (code_dco_phase_EPL(index,:)) + LocalCorreArray(index);% ��Ƭ����֧·������λ
        tmptmp = floor(mod(tmptmp2+codelength*2,codelength*2))+1;    %������Ƶ��������
    end
    for ms = 1:1:totalnum/2
        
        code_dco_phase_EPL=repmat(code_dco_phase_twocodeperiod,LengthCorreArray,1);%֧·����λ����
        for index=1:1:LengthCorreArray
            tmptmp2(index,:) = (code_dco_phase_EPL(index,:)) + LocalCorreArray(index);% ��Ƭ����֧·������λ
            tmptmp = floor(mod(tmptmp2+codelength*2,codelength*2))+1;    %������Ƶ��������
        end
        inputdata_twocodeperiod = A*getdatafromfile(fp,datanum_onecodeperiod*2)+randn(1,datanum_onecodeperiod*2)+sqrt(-1)*randn(1,datanum_onecodeperiod*2);
        %2ms��Ե����
        onemssum=inputdata_twocodeperiod.*exp(-j*2*pi*carrier_dco_phase_twocodeperiod).*codetwo(tmptmp);
        %2ms�Ĳ���������ۼ�
        %splitMatrixAndSum �ڶ�������Ϊ���ۼ������ڵķֶ�����
        rowSums = splitMatrixAndSum(onemssum, seg*2);
        for index = 1:1:LengthCorreArray
            coh(index,:) = fft(rowSums(index,:), fftNUM); % ÿ�н���K��FFT
        end
        coh(:,:)=abs(coh(:,:));
        nocoh=nocoh+coh;
        onemssum=0;
        code_dco_phase_twocodeperiod=code_dco_phase_twocodeperiod+code_dco_phase_zengliang*2;
        carrier_dco_phase_twocodeperiod=carrier_dco_phase_twocodeperiod+carrier_dco_phase_zengliang*2;
    end
    %ͳ�Ʒ�ֵ
%     figure
%     surf(nocoh)
    fidr = fopen('re.txt','a');
    %������ֵ��������λ��������Ƶ��
    [maxValue, linearIndex] = max(nocoh(:));  % �������ֵ����������
    [row, col] = ind2sub(size(nocoh), linearIndex);  % ����������ת��Ϊ�к�������
    f1=1000/(fftNUM*1/seg);%�ֱ���
    maxdop=LocalDop+(col-1)*f1;
   fprintf(fidr,'bpsk only cn0 %d   max = %f  fdop= %f Hz  codei = %05d  \n',...
        cn0,maxValue ,maxdop,row);
    fclose(fidr);
end
fclose(fp);








