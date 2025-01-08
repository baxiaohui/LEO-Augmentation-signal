%ca��Ϊ-1��1
%low����Ϊ-1��1
%high ����Ϊ0��1��������λ��1 
clear all
format long g
format compact
filename = "LEO_SIGNAL_TEST1bit.txt";
fid = fopen(filename,"w");
start=input("��������ֵ�����͹쵼����ǿ�źŷ��������");
%%%%%%% ����������֣�
%������ƵƵ��
%Frf=input("������ƵƵ��1522224000����1202025000��");
PRN=input("��������PRN�ţ�");
frq_dop=input("�����Ӧ���ز�������:");
frq_dop_acce=input("�����Ӧ���ز������ռ��ٶ�(ע������������ļ��ٶ�):");

Frf = 1522224000;                                        %%%%%%%% ��ƵƵ�� XW-L
Freq_sample = (16.368e6);                                %%%%%%%% ����Ƶ��
Fc = 1e6;                                              %%%%%%%% ����Ƶ�޶�����Ƶƫ�ز�Ƶ��
Freq_code = 2.046e6;                                       %%%%%%%% α��Ƶ��
acce = [0  0 0 0];                                         %%%%%%%% �ز������ռ��ٶ�
Freq_dop = [+0  -456   +3197  -2641];                   %%%%%%%% �ز������� 
Freq_dop(1)=frq_dop;
acce(1)=frq_dop_acce;
freq_dot = acce/3e8*Frf;                                   %%%%%%%% ���ݹ�ʽ f = v / c * Frf��Ƶ�ʱ仯�� f' = acce / c * Frf
freq_code_dot = acce/3e8*Freq_code;                        %%%%%%%% ͬ��ɵ�α��Ƶ�ʱ仯��Ϊ f_code' = acce / c * Freq_code
carrier_dco_delta = (Freq_dop+Fc)/Freq_sample;              %%%%%%%% �ز���cos(2*pi*fc*t)����ʽ���ź��У���ɢ������cos(2*pi*fc*n*Ts)
carrier_dop_to_code_dop = Frf/Freq_code;                     %�ز���������������յı�����ϵ
code_dco_delta = ((Freq_dop)/carrier_dop_to_code_dop + Freq_code)/Freq_sample; 
carrier_dco_phase = [0 0 0 0];                             %%%%%%%% �ز���λ���Ŀ�����
code_dco_phase = [0 0 0 0];                                 %%%%%%%% ����λ���Ŀ�����
mscount = zeros(1,4);                                      %%%%%%%% ������������Ŀ�����
bitnumlow = zeros(1,4);                                     %%%%%%%% ���ؼ��������Ŀ�����(���٣�250bps)
bitnumhigh = zeros(1,4);                                    %%%%%%%% ���ؼ��������Ŀ�����(���٣�3000bps)
chipcounter =zeros(1,4);
%  chipcounter(1)=10;
gt = zeros(1,4);                                             %���ٵ���ѡͨ��,0����ѡͨ��1����ѡͨ
codelow=zeros(4,2046);
codehigh_initial=zeros(4,2046);
%%�����뷢�������������������ٸ���
for i=1:1:4
    [codetemphigh,codetemplow]=CAgenerate(i);
    codehigh_initial(i,:)=codetemphigh;
    codelow(i,:)=codetemplow;
end
codehigh = codehigh_initial;%��������������Ϊ��λ��׼��
xk = zeros(1,4);   %6bitƴ�ӷ���

%���ٵ���250bps
FrameSynFlag = [ 1 1 1 0 0 0 1 0 0 1 0 0 1 1 0 1 1 1 1 0 1 0 0 0];                          %%%%%%%% ֡ͷ��24����
Tail1 = [0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,0];
Tail=[];
for i=1:1:11
    Tail=[Tail Tail1]; %726bit
end

BITlow = [];                                                                              
for i=1:1:100
    BITlow = [BITlow FrameSynFlag  Tail]; %750bit�ظ�100�Σ���300������
end
BITlow = BITlow*2-1; %3S*100 ,100֡

%���ٵ��� 3000bps
Tail=[0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,...
    0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1];
BIThigh = [];
temp=[];
for i=1:32   
    temp=[temp Tail];     %93*32=2976�����ٷ���
    
end

for i=1:1:300                           
    BIThigh = [BIThigh FrameSynFlag temp];  %����300��  ����0��1����-1��1
end
 

sigma = 1;                                           
th = 1; 
% A(1) = sigma* sqrt(2*10^(4.5)/Freq_sample);
% A(2) = sigma* sqrt(2*10^(4.8)/Freq_sample);
% A(3) = sigma* sqrt(2*10^(4.4)/Freq_sample);
% A(4) = sigma* sqrt(2*10^(4)/Freq_sample);
A=ones(1,4);
n = 0;
flag = 0;
num = 0;
samplenum = 0;
sec = 0;




while (1)
    n = n + 1;
   
    
    for i = 1:4          
        carrier_dco_phase(i) = carrier_dco_phase(i) + carrier_dco_delta(i) + 0.5*freq_dot(i)*(2*n-1)/Freq_sample/Freq_sample;
        code_dco_phase(i) = code_dco_phase(i)+ code_dco_delta(i) + 0.5*freq_code_dot(i)*(2*n-1)/Freq_sample/Freq_sample;
        if(carrier_dco_phase(i) >= 1)
            carrier_dco_phase(i) = carrier_dco_phase(i) - 1;
        end 
        if(code_dco_phase(i) >= 1)
            code_dco_phase(i) = code_dco_phase(i) - 1;
            chipcounter(i) = chipcounter(i) + 1;
            if(chipcounter(i) >= 2046)
                chipcounter(i) = 0;
                mscount(i) = mscount(i) + 1;
                if(mscount(i) == 4)
                    bitnumlow(i) = bitnumlow(i) + 1;
                    mscount(i) = 0;
                    gt(i) = 0;
                elseif(mscount(i) == 1)
                    xk(i) = 32*BIThigh(bitnumhigh(i)+1)+16*BIThigh(bitnumhigh(i)+2)+8*BIThigh(bitnumhigh(i)+3)+4*BIThigh(bitnumhigh(i)+4)+...
                            2*BIThigh(bitnumhigh(i)+5)+BIThigh(bitnumhigh(i)+6);%ע��BITHIGHֵΪ0��1                       
                    codehigh(i,:) = circshift(codehigh_initial(i,:),xk(i));  %���CA���Ƿ���λ������             
                    bitnumhigh(i) = bitnumhigh(i) + 6;
                    gt(i) = 1;
                elseif(mscount(i) == 2)
                %����ÿ�����ŷ�����������Ƶ�����ڣ�bitnumlow����Ҫ+1
                    gt(i) = 0;
                elseif(mscount(i) == 3)  
                    xk(i) = 32*BIThigh(bitnumhigh(i)+1)+16*BIThigh(bitnumhigh(i)+2)+8*BIThigh(bitnumhigh(i)+3)+4*BIThigh(bitnumhigh(i)+4)+...
                            2*BIThigh(bitnumhigh(i)+5)+BIThigh(bitnumhigh(i)+6);%ע��BITHIGHֵΪ0��1                       
                    codehigh(i,:) = circshift(codehigh_initial(i,:),xk(i));  %���CA���Ƿ���λ������        
                    bitnumhigh(i) = bitnumhigh(i) + 6;
                    gt(i) = 1;
                else
                    disp('err');
                    pause;
                end
            end
        end
    end

%     result = randn*sigma + sqrt(-1)*randn*sigma; %%%%%%%% ����IQ��·�ź�����
      result = 0 + sqrt(-1)*0; %%%%%%%% ����IQ��·�ź�����
    for ss = 1:4
    %codehigh�����뵼���������
        result = result + A(ss)*codelow(ss,chipcounter(ss)+1)*(1-gt(ss))*BITlow(bitnumlow(ss)+1)*exp(sqrt(-1)*2*pi*carrier_dco_phase(ss))+...
            A(ss)*codehigh(ss,chipcounter(ss)+1)*gt(ss)*exp(sqrt(-1)*2*pi*carrier_dco_phase(ss));%���ٵ����Լ�ӳ����CA���
        
        
            
    end
    result_output = result;

    if(real(result_output)>th)    %%%%%%%% �ж�ʵ�� ����������
        afterquani = 3;
    elseif (real(result_output)>0)
        afterquani = 1;
    elseif(real(result_output)>-th)
        afterquani = -1;
    else
        afterquani = -3;
    end


    %%%%%%%% q·�ź�ͬi·�ź�����
    if(imag(result_output)>th)   %%%%%%%% �о���ʽͬi·�ź�
        afterquanq = 3;
    elseif (imag(result_output)>0)
        afterquanq = 1;
    elseif(imag(result_output)>-th)
        afterquanq = -1;
    else
        afterquanq = -3;
    end
    %1bit����
    if(imag(result_output)>0)
        afterquanq=1;
    else
        afterquanq=-1;
    end
    if(real(result_output)>0)
        afterquani=1;
    else
        afterquani=-1;
    end
    fprintf(fid,'%d %d\n',round(real(result)),round(imag(result)));
    samplenum = samplenum + 1;   
        
    
    
    
%    fprintf(fid,'%f %f\n',real(result),imag(result));
%     samplenum = samplenum + 1;   

%     signedData = int16([afterquani, afterquanq]);  % i q ��ռ16λ  
%     fileID = fopen(filename, 'a');  
%     fwrite(fileID, signedData, 'int16');    
%     fclose(fileID);   
    
    if(samplenum == Freq_sample*0.001)
        samplenum = 0;
        sec = sec + 1;
        fprintf('%d ms....\n',sec);
    end    
    if(sec == 1001)
        break;
    end
    
end
fclose all;
