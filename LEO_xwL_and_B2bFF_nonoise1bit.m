%ca码为-1与1
%low电文为-1与1
%high 电文为0与1（方便移位）1 
clear all
format long g
format compact
filename = "LEO_SIGNAL_TEST1bit.txt";
fid = fopen(filename,"w");
start=input("输入任意值后开启低轨导航增强信号发射软件：");
%%%%%%% 定义参数部分：
%输入射频频率
%Frf=input("输入射频频率1522224000或者1202025000：");
PRN=input("输入卫星PRN号：");
frq_dop=input("输入对应的载波多普勒:");
frq_dop_acce=input("输入对应的载波多普勒加速度(注意这里是载体的加速度):");

Frf = 1522224000;                                        %%%%%%%% 射频频率 XW-L
Freq_sample = (16.368e6);                                %%%%%%%% 采样频率
Fc = 1e6;                                              %%%%%%%% 低中频无多普勒频偏载波频率
Freq_code = 2.046e6;                                       %%%%%%%% 伪码频率
acce = [0  0 0 0];                                         %%%%%%%% 载波多普勒加速度
Freq_dop = [+0  -456   +3197  -2641];                   %%%%%%%% 载波多普勒 
Freq_dop(1)=frq_dop;
acce(1)=frq_dop_acce;
freq_dot = acce/3e8*Frf;                                   %%%%%%%% 根据公式 f = v / c * Frf得频率变化率 f' = acce / c * Frf
freq_code_dot = acce/3e8*Freq_code;                        %%%%%%%% 同理可得伪码频率变化率为 f_code' = acce / c * Freq_code
carrier_dco_delta = (Freq_dop+Fc)/Freq_sample;              %%%%%%%% 载波以cos(2*pi*fc*t)的形式在信号中，离散化后变成cos(2*pi*fc*n*Ts)
carrier_dop_to_code_dop = Frf/Freq_code;                     %载波多普勒与码多普勒的比例关系
code_dco_delta = ((Freq_dop)/carrier_dop_to_code_dop + Freq_code)/Freq_sample; 
carrier_dco_phase = [0 0 0 0];                             %%%%%%%% 载波相位，四颗卫星
code_dco_phase = [0 0 0 0];                                 %%%%%%%% 码相位，四颗卫星
mscount = zeros(1,4);                                      %%%%%%%% 毫秒计数器，四颗卫星
bitnumlow = zeros(1,4);                                     %%%%%%%% 比特计数器，四颗卫星(低速：250bps)
bitnumhigh = zeros(1,4);                                    %%%%%%%% 比特计数器，四颗卫星(高速：3000bps)
chipcounter =zeros(1,4);
%  chipcounter(1)=10;
gt = zeros(1,4);                                             %高速低速选通器,0低速选通，1高速选通
codelow=zeros(4,2046);
codehigh_initial=zeros(4,2046);
%%调用码发生器函数，两个，低速高速
for i=1:1:4
    [codetemphigh,codetemplow]=CAgenerate(i);
    codehigh_initial(i,:)=codetemphigh;
    codelow(i,:)=codetemplow;
end
codehigh = codehigh_initial;%高速码设两个，为移位做准备
xk = zeros(1,4);   %6bit拼接符号

%低速电文250bps
FrameSynFlag = [ 1 1 1 0 0 0 1 0 0 1 0 0 1 1 0 1 1 1 1 0 1 0 0 0];                          %%%%%%%% 帧头，24比特
Tail1 = [0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,0];
Tail=[];
for i=1:1:11
    Tail=[Tail Tail1]; %726bit
end

BITlow = [];                                                                              
for i=1:1:100
    BITlow = [BITlow FrameSynFlag  Tail]; %750bit重复100次，即300秒数据
end
BITlow = BITlow*2-1; %3S*100 ,100帧

%高速电文 3000bps
Tail=[0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,...
    0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1];
BIThigh = [];
temp=[];
for i=1:32   
    temp=[temp Tail];     %93*32=2976个高速符号
    
end

for i=1:1:300                           
    BIThigh = [BIThigh FrameSynFlag temp];  %共发300秒  ，发0和1，非-1与1
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
                            2*BIThigh(bitnumhigh(i)+5)+BIThigh(bitnumhigh(i)+6);%注意BITHIGH值为0或1                       
                    codehigh(i,:) = circshift(codehigh_initial(i,:),xk(i));  %检查CA码是否移位！！！             
                    bitnumhigh(i) = bitnumhigh(i) + 6;
                    gt(i) = 1;
                elseif(mscount(i) == 2)
                %低速每个符号发两个低速扩频码周期，bitnumlow不需要+1
                    gt(i) = 0;
                elseif(mscount(i) == 3)  
                    xk(i) = 32*BIThigh(bitnumhigh(i)+1)+16*BIThigh(bitnumhigh(i)+2)+8*BIThigh(bitnumhigh(i)+3)+4*BIThigh(bitnumhigh(i)+4)+...
                            2*BIThigh(bitnumhigh(i)+5)+BIThigh(bitnumhigh(i)+6);%注意BITHIGH值为0或1                       
                    codehigh(i,:) = circshift(codehigh_initial(i,:),xk(i));  %检查CA码是否移位！！！        
                    bitnumhigh(i) = bitnumhigh(i) + 6;
                    gt(i) = 1;
                else
                    disp('err');
                    pause;
                end
            end
        end
    end

%     result = randn*sigma + sqrt(-1)*randn*sigma; %%%%%%%% 生成IQ两路信号噪声
      result = 0 + sqrt(-1)*0; %%%%%%%% 生成IQ两路信号噪声
    for ss = 1:4
    %codehigh内容与导航电文相关
        result = result + A(ss)*codelow(ss,chipcounter(ss)+1)*(1-gt(ss))*BITlow(bitnumlow(ss)+1)*exp(sqrt(-1)*2*pi*carrier_dco_phase(ss))+...
            A(ss)*codehigh(ss,chipcounter(ss)+1)*gt(ss)*exp(sqrt(-1)*2*pi*carrier_dco_phase(ss));%高速电文以及映射至CA码里。
        
        
            
    end
    result_output = result;

    if(real(result_output)>th)    %%%%%%%% 判断实部 两比特量化
        afterquani = 3;
    elseif (real(result_output)>0)
        afterquani = 1;
    elseif(real(result_output)>-th)
        afterquani = -1;
    else
        afterquani = -3;
    end


    %%%%%%%% q路信号同i路信号量化
    if(imag(result_output)>th)   %%%%%%%% 判决方式同i路信号
        afterquanq = 3;
    elseif (imag(result_output)>0)
        afterquanq = 1;
    elseif(imag(result_output)>-th)
        afterquanq = -1;
    else
        afterquanq = -3;
    end
    %1bit量化
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

%     signedData = int16([afterquani, afterquanq]);  % i q 各占16位  
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
