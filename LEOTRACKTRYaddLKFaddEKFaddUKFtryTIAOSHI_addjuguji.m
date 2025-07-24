clear all
%���ò���
runtime = 100;                                             %��������ʱ�䣬��λ��
filename = 'LKF.txt';                                       %��ӡ���ļ�
format long
%%%%%%% ����������֣�
speedoflight = 299792458;
Freq_sample = 16.369e6;                                     %%%%%%%%����Ƶ��
Fc = 4092000;                                              %%%%%%%% ����Ƶ�޶�����Ƶƫ�ز�Ƶ��
Freq_code = 2.046e6;                                       %%%%%%%% ��Ƶ��Ƶ��
acce = [100  0 0 0];                                         %%%%%%%% ������ٶ�
jerk=[0 0 0 0];
Freq_IF = [Fc+10000  Fc+1000   Fc+2500  Fc-1000];           %%%%%%%% ʵ�ʵ���Ƶ�ز�Ƶ��
Frf = 1522224000;                                          %%%%%%%% ��ƵƵ��
codedopcoef = Frf/Freq_code;                               %��ƵƵ������Ƶ��Ƶ��֮��
codelength = 2046;                                         %�볤
bitperiod = 4;                                            %bit���ڣ���λms
freq_dot = acce/speedoflight*Frf;                                   %%%%%%%% ���ݹ�ʽ f = v / c * Frf��Ƶ�ʱ仯�� f' = acce / c * Frf
freq_code_dot = acce/speedoflight*Freq_code;                        %%%%%%%% ͬ��ɵ�α��Ƶ�ʱ仯��Ϊ f_code' = acce / c * Freq_code
freq_dot_2rd=jerk/speedoflight*Frf;                                  %���ױ仯��
freq_code_dot_2rd=jerk/speedoflight*Freq_code;  
chipcounter = [0 0 0 0];                                   %%%%%%%% ��Ƭ���������Ŀ�����
carrier_dco_delta = (Freq_IF)/Freq_sample;                %%%%%%%% �ز���cos(2*pi*fc*t)����ʽ���ź��У���ɢ������cos(2*pi*fc*n*Ts),��������Ƶ�ź���Ϊ(Freq_IF - Fc) / Freq_sample
code_dco_delta = ((Freq_IF-Fc)/codedopcoef+Freq_code)/Freq_sample;%%%%%%%% ͬ��
carrier_dco_phase = [0 0 0 0];                             %%%%%%%% ��ʼ���ز���λ���Ŀ�����
code_dco_phase=[0 0 0 0];                                  %%%%%%%% ��ʼ��α����λ���Ŀ�����
mscount = zeros(1,4);                                      %%%%%%%% ������������Ŀ�����

bitnumlow = zeros(1,4);                                     %%%%%%%% ���ؼ��������Ŀ�����(���٣�250bps)
bitnumhigh = zeros(1,4);                                    %%%%%%%% ���ؼ��������Ŀ�����(���٣�3000bps)
gt = zeros(1,4);                                             %���ٵ���ѡͨ��,0����ѡͨ��1����ѡͨ


%%%%%%%% �����뷢�������������Ŀ����ǵ�α��
codelow=zeros(4,2046);
codehigh_initial=zeros(4,2046);
%%�����뷢�������������������ٸ���
for i=1:1:4
    [codetemphigh,codetemplow]=CAgenerate(i);
    codehigh_initial(i,:)=codetemphigh;
    codelow(i,:)=codetemplow;
end

%%%%%%%% �����о�ǰͨ��һ����ͨ�˲������������£�
filterorder = 21;                                                  %%%%%%%% ����Ϊ20���˴���ʾ��21����
result_input = zeros(1,filterorder);                               %%%%%%%% ͨ���˲����������������飬���ȵ���filterorder
%%% ���ɵ�ͨ�˲�����ѡ��Kaiser��FIR�˲�����BetaֵΪ0.5������Ϊ20��������30.69MHz����ֹƵ��1.1MHz��ϵ�����£�
% lowpassfilter = [0.021296855511562 0.0276752982618497 0.0341164198702814 0.0404241078806896 0.0463987613744401...
%                 0.051845537789563 0.0565825791873747 0.060448817384489 0.0633109732940896 0.0650694024346712...
%                 0.0656624940219796 0.0650694024346712 0.0633109732940896 0.060448817384489 0.0565825791873747...
%                 0.051845537789563 0.0463987613744401 0.0404241078806896 0.0341164198702814 0.0276752982618497...
%                 0.021296855511562
%                 ];                               %%%%%%%% 1*21���˲���ϵ��
%%% ���ɵ�ͨ�˲�����ѡ��Kaiser��FIR�˲�����BetaֵΪ0.5������Ϊ20��������16.368MHz����ֹƵ��1.1MHz��ϵ�����£�
lowpassfilter = [  -0.024878413973009  -0.019403240492844  -0.008442779911688   0.007679140518129...
   0.027927884076498   0.050642413108743   0.073714613080674   0.094841094821135...
   0.111813064702503   0.122803007209291   0.126606433721137   0.122803007209291...
   0.111813064702503   0.094841094821135   0.073714613080674   0.050642413108743...
   0.027927884076498   0.007679140518129  -0.008442779911688  -0.019403240492844...
  -0.024878413973009
                ];                               %%%%%%%% 1*21���˲���ϵ��

%%% ���˲���ϵ����һ����ʹ������ͨ���˲������ֵ����
H = sum(lowpassfilter.*lowpassfilter);
lowpassfilter = lowpassfilter/sqrt(H);
%% ��������
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
for i=1:1:1000/2
    BITlow = [BITlow FrameSynFlag  Tail]; %750bit�ظ�100�Σ���300������ for i=1:1:100
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

for i=1:1:3000/3                           
    BIThigh = [BIThigh FrameSynFlag temp];  %����300��  ����0��1����-1��1  for i=1:1:300 GAIFA1000S
end
%%%%%%%% �趨�о����ޣ�
sigma = 1;                                           
th = 1; 
BITHIGHTRANS=BIThightrans(BIThigh);
%%%%%%%% ���ݹ�ʽ��(1)sigma^2 =  *N0/2; (2)sigma0 = N0*B;
%%%%%%%% (3)CNR - 10*lgB = 10*lg(A^2/2/sigma0^2) ��
%%%%%%%% A = sigma * sqrt(2 * 10^(CN0/10)/fs)   ********
CN0 = 34
A(1) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%
A(2) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%
A(3) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%
A(4) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%

n = 0;
flag = 0;
num = 0;                 
samplenum = 0;    
sec = 0; 


%���ٲ��������Լ��ۼ������ʼ��
mode = 2;                                          %�趨ģʽ . ==0 pll   ==1 fll   ==2 LKF   ==3 EKF�������   ==4 EKF���������  ==5 �����κε���������CSK���  ==6 UKF�������
                                                   %mode==8  LKF 1MSԤ��2ms����
                                                   
AcqSatNUM = 1;                                     %�趨Ҫ���ٵ�����PRN���
LocalDop = Freq_IF(AcqSatNUM)-Fc+20;                %��ʼ�ز������գ�ע��Ҫ���趨ֵ�ȽϽӽ�
[codetemphigh,codetemplow]=CAgenerate(AcqSatNUM);                           
LocalCode = codetemplow;                                       %��ʼ����ͨ����Ƶ��
for cskn=0:1:63
    Localhightraverse(cskn+1,:)=circshift(codetemphigh,cskn);
end

code_dco_phaseL = 0;                               %���ظ�������λ
carrier_dco_phaseL = 0;                            %���ظ�������λ
% code_dco_phaseL2 = 0 ;%���ظ�������λ������һ����0-1������û���õ�����
code_dco_deltaL = ((LocalDop)/codedopcoef+Freq_code)/Freq_sample;     %������Ƶ�ʲ���
carrier_dco_deltaL = (Fc + LocalDop)/Freq_sample;                     %�����ز�Ƶ�ʲ���
chipcntL = 0;                                     %������Ƭ����
mscountL = 0;                                     %����ms����
bitnumL = 0;                                      %����bit����
mstime=0;                                         %����ͳ�Ƹ������
LocalCorreArray = [-1.5:0.1:1.5];                 %������������Ƭ��Χ����Ƭ���
LengthCorreArray = length(LocalCorreArray);       %�������Ŀ
mid = (LengthCorreArray+1)/2;                     %P·����
corrsum = zeros(1,LengthCorreArray);              %����֧·�ۼ�ֵ
corrcohsum_carr = zeros(1,LengthCorreArray);      %����ز����ۼ�ֵ
corrcohsum_code = zeros(1,LengthCorreArray);      %����뻷�ۼ�ֵ
corrnoncohsum_code = zeros(1,LengthCorreArray);   %������뻷�ۼ�ֵ
% corrcohsum_carr_FLL=zeros(1,LengthCorreArray);    %FLL�ۼ�ֵ
corrcohsum_carr_FLL_first=0;                               %�����һ��FLL���ۼ�ֵ
corrcohsum_carr_FLL_second=0;                               %������ϴ�FLL���ۼ�ֵ
dump = 0;                                         %Ϊ1ʱ����ʾһ�����ڵ��ۼӽ�����һ�����һ�����ڱ�ʾ1����Ƶ�����ڣ�B1C������
localn = 0;                                       %���ز��������
FLL_bitcount=0;                                        %FLL���ڼ�¼��ǰ��ɵ��ڼ�����������
%%
%�뻷·��������
%%%%%%%%%%%�����뻷����%%%%%%%%%%%%%
T = 0.001;                                        %��λΪ�룬��ʾ1ms����һ��
cd = 0;                                           %�������ֵ
cd_1 = 0;                                         %���������һ��ʱ�̵�ֵ
vo = 0;                                           %ѹ���������
vo_1 = 0;                                         %ѹ��������һ��ʱ�̵�ֵ
codeloop_cohtime = 4;                             %�뻷���ʱ�䣬2��ʾ2���ۼ����ڣ�һ����2ms��Ҫ�������
codeloop_noncohtime = 2;                         %�뻷������ۼӴ���
codeloop_noncohcnt = 0;                           %�뻷������ۼӼ�����
bn_code = 1;                                      %�뻷����1
cd1 = 8/3*bn_code;                                %�����뻷����һ
cd2 = cd1*cd1/2*T*codeloop_cohtime*codeloop_noncohtime/2;   %�����뻷��������K2
%%
%�ز���·PLL��������
%%%%%%%%%%%%%%%%%%%%%%�����ز�������%%%%%%%%%%%%%%%%%%%%%%%
carrierlooporder = 3;                             %�ز����������ã�2��ʾ2�׻�·��3��ʾ3�׻�·
pd = 0;                                           %�ز����������
pd_1 = 0;                                         %��һ�μ��������
pd_2 = 0;                                         %����һ�εļ��������
pd_3=0;
vn = 0;                                           %ѹ���������
vn_1 = 0;                                         %��һ��ѹ���������
vn_2 = 0;                                         %����һ��ѹ���������
vn_3=0;
carrierloop_cohtime = 2;                          %��·��ɻ���ʱ�䣬��λms
bn_carr = 5;                                     %�ز���·����
%����
cp1_2nd = 8/3*bn_carr;                            %���׻�·ϵ��1
cp2_2nd = cp1_2nd*cp1_2nd/2;                      %���׻�·ϵ��2
%����
cp1_3rd = 60/23*bn_carr;                          %���׻�·ϵ��1
cp2_3rd = cp1_3rd*cp1_3rd*4/9;                    %���׻�·ϵ��2
cp3_3rd = (cp1_3rd*cp1_3rd*cp1_3rd)*2.0/27.0;     %���׻�·ϵ��3
%�Ľ�
cp1_4th=64/27*bn_carr;
cp2_4th=(cp1_4th^2/2);
cp3_4th=(cp1_4th^3/8);
cp4_4th=(cp1_4th^4/64);

%%
%�ز���·FLL��������
%%%%%%%%%%%%%%%%%%%%%%������Ƶ������%%%%%%%%%%%%%%%%%%%%%%%
carrierloopFLLorder = 2;                          %��Ƶ���������ã�2��ʾ2�׻�·��3��ʾ3�׻�·
fd = 0;                                           %�ز����������
fd_1 = 0;                                         %��һ�μ��������
fd_2 = 0;                                         %����һ�εļ��������
fd_3=0;
fo_n = 0;                                           %ѹ���������
fo_n_1 = 0;                                         %��һ��ѹ���������
fo_n_2 = 0;                                         %����һ��ѹ���������
fo_n_3 = 0;                                         %����һ��Ƶ�����
carrierloop_fll_cohtime = 4;                       %FLL��·����ɻ���ʱ��ȡ1���������ڣ���λms
carrierloop_fll_real_cohtime=carrierloop_fll_cohtime/2; %FLL������ɻ���ʱ�� ��Ҫ�����������ֵ�趨��ʼ�ز�������ƫ�����
%1�����ı��ص����ڷֳ�ǰһ��ͺ�һ�룬�൱�ڵ��λ���ʱ��Ϊcarrierloop_fll_cohtime/2
%atan�ļ���ΧΪ��1/(4*carrierloop_fll_cohtime/2) Hz
%��Ϊ��1/(4*carrierloop_fll_real_cohtime)Hz
carrierloop_fll_noncohcnt = 0;                       %��ǰ���˼������ڵĵ��Ľ������
carrierloop_fll_noncohtime =2;                      %�ܼ�ʹ�ü����������ڽ������
bn_carr_fll =3;                                     %�ز���·���� 38dbHZʱbn10���Ը��� 5
cross_dot=0;                                         %��������ʼ��
cf1_1st = 4.0 * bn_carr_fll * T * carrierloop_fll_cohtime*carrierloop_fll_noncohtime/2.0;
%2��FLLϵ��
c1 = 8.0/3.0 * bn_carr_fll;
c2 = (c1*c1)/2.0;
cf1_2nd = c1 * (T*carrierloop_fll_cohtime*carrierloop_fll_noncohtime)/2.0;
cf2_2nd = c2 * ((T*carrierloop_fll_cohtime*carrierloop_fll_noncohtime)/2.0)^2;
%3��FLLϵ��
c1 = 60.0/23.0 * bn_carr_fll;
c2 = (c1*c1)*4.0/9.0;
c3 = (c1*c1*c1)*2.0/27.0;
cf1_3rd = c1 * (T*carrierloop_fll_cohtime*carrierloop_fll_noncohtime)/2.0;
cf2_3rd = c2 * ((T*carrierloop_fll_cohtime*carrierloop_fll_noncohtime)/2.0)^2;
cf3_3rd = c3 * ((T*carrierloop_fll_cohtime*carrierloop_fll_noncohtime)/2.0)^3;

%%
%���������ʼ��
FCODE = [];                %ͳ����Ƶ��Ƶ��
FCARR = [];                %ͳ���ز�Ƶ��
Earr = [];                 %ͳ��E·
Parr=[];                   %ͳ��P·
Larr=[];                   %ͳ��L·
loopcnt = 0;               %��·����
cntL = 0;                  %δʹ��
accum = [];                %�洢ÿ�����ۼӽ�����ڼ���NP��PLD
NParray = [];              %�洢ÿ20ms�õ���NP
PLD = [];                  %ͳ��PLD ��FLLʹ��ʱ������λ��PLD���ܲ�Ϊ1
PD = [];                   %ͳ��PLL������
PDFLL=[];                  %ͳ��FLL������
CN0L=[];                    %CN0
codeerr = [];              %ͳ�Ƹ��ٵ���Ƶ������λ�����ɵ���Ƶ������λ֮��
carrerr = [];              %ͳ�Ƹ��ٵ��ز���λ�����ɵ��ز���λ֮��
DOPERRORre=[];             %ͳ�Ƹ��ٵ��ز������ɵ��ز�֮Ƶ�ʲ�
flag = 1;                  %Ϊ1��ʾʹ����λ��չ
testopen = 1;
M2=0;%����������ȹ���
M4=0;
corrsumforcskdemodulate=zeros(1,64); %����CSK���
highbitcsknoco=[];%����ɽ�����ڴ��CSK����
highbitcskco=[];%��ɽ��
highbitcsknoco_symbol=[];
highbitcskco_symbol=[];

FCARRfordopcomp=[];%���ڶ����ղ���
avgold=LocalDop;%���ڶ����ղ���
avg=LocalDop;
rng(1);%�趨��������ӣ�ʹ�������������ͬ ����ɾ�� 
% rng(2);
%  rng(3);
accf=[];
acccom=0;
n1=0;
avgtime=1000;
%�ź����ɼ�����
while (1)
    %%
    %�ò���Ϊ�źŸ���PLL+FLL+�뻷·���࣬����ģ��
    if(dump == 1)
        loopcnt = loopcnt + 1
        dump = 0;
        cntL = cntL + 1;
        
        if(testopen) 
            Earr = [Earr corrsum(mid-5)];
            Larr = [Larr corrsum(mid+5)];
            Parr = [Parr corrsum(mid+0)];
        end
        %ͳ��CSK�����ѧ �ж������� mstime/2==0? 
        if(mod(mstime,2)==0)
            %����ɽ�� 
            [maxValue, maxIndex] = max(abs(corrsumforcskdemodulate));
            binaryArray1 = double(dec2bin(maxIndex-1,6)) - '0';
            highbitcsknoco=[highbitcsknoco binaryArray1];
            
            highbitcsknoco_symbol=[highbitcsknoco_symbol (maxIndex-1)];
            %��ɽ��
            [maxValue, maxIndex] = max((real(corrsumforcskdemodulate)));%��realȡ����ֵ����ֹ�ز���λ����180���ģ��
            binaryArray2 = double(dec2bin(maxIndex-1,6)) - '0';
            highbitcskco=[highbitcskco binaryArray2];
            
            highbitcskco_symbol=[highbitcskco_symbol (maxIndex-1)];
            corrsumforcskdemodulate=zeros(1,64);
        end
        
        
        corrcohsum_carr = corrcohsum_carr + corrsum;         %����ۼ�Ϊ���ز�����
        corrcohsum_code = corrcohsum_code + corrsum;         %����ۼ�Ϊ���뻷����
        if(mscountL == 0)                                    %
            mscountL1_bitperiod = bitperiod;
        else
            mscountL1_bitperiod = mscountL;
        end
%         accum(mscountL1_bitperiod) = corrsum(mid+0);           %accum������������Ⱥ�PLD  ����� 
        M4=M4+(real(corrsum(mid+0))^2+imag(corrsum(mid+0))^2)^2;
        M2=M2+real(corrsum(mid+0))^2+imag(corrsum(mid+0))^2;
      
        %ʹ��FLLʱ�����λ���ʱ���൱��carrierloop_fll_cohtime/2
        if(mscountL1_bitperiod<=bitperiod/2)
             corrcohsum_carr_FLL_first=corrcohsum_carr_FLL_first+corrsum(mid+0); %ΪFLL������  ÿһ���������ڵ�ǰһ��
        else
             corrcohsum_carr_FLL_second=corrcohsum_carr_FLL_second+corrsum(mid+0);%ΪFLL������  ÿһ���������ڵĺ�һ��       
        end
        corrsum = zeros(1,LengthCorreArray);                   %ÿms�ۼ�ֵ����
        if(mode == 0)
        %PLL
            if( mod(mscountL1_bitperiod,carrierloop_cohtime) == 0)  %�ز���·����
                P = corrcohsum_carr(mid);                     %ʹ��P·
                pd = atan(imag(P)/real(P))/2/pi;              %������
                PD=[PD pd];
                delta_pd_n1 = pd - pd_1;                     %�������໷ʹ��
                delta_pd_n2 = pd_1 - pd_2;                   %�������໷ʹ��
                delta_pd_n3 = pd_2 - pd_3;
                if(flag == 1)
                    if(abs(delta_pd_n1) >= 0.25)
                        if(delta_pd_n1 > 0)
                            delta_pd_n1 = (delta_pd_n1-0.5);
                        else
                            delta_pd_n1 = (delta_pd_n1+0.5);
                        end
                    end
                    if(abs(delta_pd_n2) >= 0.25)
                        if(delta_pd_n2 > 0)
                            delta_pd_n2 = (delta_pd_n2-0.5);
                        else
                            delta_pd_n2 = (delta_pd_n2+0.5);
                        end
                    end
                    
                     if(abs(delta_pd_n3) >= 0.25)
                        if(delta_pd_n3 > 0)
                            delta_pd_n3 = (delta_pd_n3-0.5);
                        else
                            delta_pd_n3 = (delta_pd_n3+0.5);
                        end
                    end
                    
                end
                if(carrierlooporder == 2)
                    vn = vn_1 + cp1_2nd * (delta_pd_n1) + cp2_2nd*T*carrierloop_cohtime/2*(pd + pd_1);
                elseif(carrierlooporder == 3)
                    vn = 2*vn_1 - vn_2 + cp1_3rd * (delta_pd_n1-delta_pd_n2) + ...
                        cp2_3rd *T*carrierloop_cohtime/2* (delta_pd_n1+delta_pd_n2)	+ cp3_3rd *(T*carrierloop_cohtime/2)^2* (pd + 2*pd_1 + pd_2);
                elseif(carrierlooporder == 4)
                    vn = 3*vn_1 - 3*vn_2 + vn_3  + cp1_4th * (delta_pd_n1-2*delta_pd_n2+delta_pd_n3 ) + ...%pd-3pd_1+3pd_2-pd_3
                        cp2_4th *(T*carrierloop_cohtime/2)* (delta_pd_n1-delta_pd_n3)	+ ...                                           %pd-pd_1-pd_2+pd_3
                        cp3_4th *(T*carrierloop_cohtime/2)^2* (delta_pd_n1+delta_pd_n3+2*delta_pd_n2 )+ ...                             %pd+pd_1-pd_2-pd_3
                        cp4_4th *(T*carrierloop_cohtime/2)^3* (pd+3*pd_1+3*pd_2+pd_3);
                    
                end
                pll_lp_out = vn - vn_1;
                carrier_dco_deltaL = carrier_dco_deltaL + pll_lp_out/Freq_sample;
                FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
    %             carrier_dco_deltaL*Freq_sample-Fc
                vn_3 = vn_2;
                vn_2 = vn_1;
                vn_1 = vn;
                pd_3 = pd_2;
                pd_2 = pd_1;
                pd_1 = pd;
                corrcohsum_carr = zeros(1,LengthCorreArray);
                
            end
        %FLL
        elseif(mode==1)
            if( mod(mscountL1_bitperiod,carrierloop_fll_cohtime) == 0)                
                %FLL���� ʹ�ö����޷����� ÿ��������������һ��dot��CRoss
                carrierloop_fll_noncohcnt = carrierloop_fll_noncohcnt + 1;
                cross_dot = cross_dot + conj(corrcohsum_carr_FLL_first)*corrcohsum_carr_FLL_second;
                if(carrierloop_fll_noncohcnt >= carrierloop_fll_noncohtime)                 
                   % fd = atan(imag(cross_dot)/real(cross_dot))/(T*carrierloop_fll_real_cohtime*2*pi);
                    
                    fd = atan2(imag(cross_dot),real(cross_dot))/(T*carrierloop_fll_real_cohtime*2*pi);%ʹ��4���޷����� 
%                     fd=carrierloop_fll_noncohtime*fd;
                    cross_dot = 0;
                    carrierloop_fll_noncohcnt = 0;                                        
                    PDFLL=[PDFLL fd];
                    %���ݽ����ı�NCO
                    if(carrierloopFLLorder == 1)
                        fo_n = fo_n_1 + cf1_1st * (fd + fd_1);
                    elseif(carrierloopFLLorder == 2)
                        fo_n=   2*fo_n_1-fo_n_2+cf1_2nd*(fd-fd_2)+cf2_2nd*(fd+2*fd_1+fd_2);
                    elseif(carrierloopFLLorder == 3)
                        fo_n = 3*fo_n_1 - 3*fo_n_2 + fo_n_3+ cf1_3rd * (fd - fd_1 - fd_2 + fd_3)+ cf2_3rd * (fd + fd_1 - fd_2 - fd_3)+ cf3_3rd * (fd + 3*fd_1 + 3*fd_2 + fd_3);
                    end
                    carrier_dco_deltaL = carrier_dco_deltaL + (fo_n-fo_n_1)/Freq_sample;
                    FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
                    fd_3 = fd_2;
                    fd_2 = fd_1;
                    fd_1 = fd;
                    fo_n_3 = fo_n_2;
                    fo_n_2 = fo_n_1;
                    fo_n_1 = fo_n;
                    
                end
                corrcohsum_carr_FLL_first = 0;
                corrcohsum_carr_FLL_second = 0;
            end
        elseif(mode==2)
            if(mod(mstime,2)==1)
                %% �������� ������mode==2 �������˲���֧ ��������
                
                % ���� ��һ�ν���ʱ��һ�γ�ʼ�� ����
                if ~exist('kf_initialized','var') || ~kf_initialized
                    % ʱ�䲽����ͬ��·�����ʣ�
                    T_kf = 2*1e-3;
                    T_coh=1*1e-3;
                    %xk=��*xk-1+B*G
                    % ״̬ת�ƾ��� �� (4��4)
                    Phi_kf = [1, T_kf, T_kf^2/2, T_kf^3/6;
                        0,     1,      T_kf,    T_kf^2/2;
                        0,     0,        1,         T_kf;
                        0,     0,        0,           1];
                    %
                    B_kf=[-1 ,-T_kf;
                           0,0 ;
                           0,0;
                           0,0;];
                    %G��ӿ����� ��deltaagnel��λ�������==Xk-1(1),�Լ�f_nco��k-1��
                    f_nco=LocalDop;%f_nco��λ��Ƶ��
                    deltaagnel=0;
                    % ��������ǿ�� ��?�� A_max��jerk PSD ��λ ����^2/s^5              
                    A_max = 500*10;
                    sigma_a = (A_max) * T_kf ;
                    Q_kf = sigma_a * [T_kf^6/252, T_kf^5/72, T_kf^4/30, T_kf^3/24;
                        T_kf^5/72, T_kf^4/20,  T_kf^3/8, T_kf^2/6;
                        T_kf^4/30,  T_kf^3/8,    T_kf^2/3,    T_kf/2;
                        T_kf^3/24,  T_kf^2/6,    T_kf/2,       1];
                   
                    % �۲���� H 
                    H_kf = [1, T_kf*1/2, T_kf^2/6, T_kf^3/24];                    
                    
                    % �������� R                
                    %R_kf=simga0^2/Ncoh=simga0^2/(Tcoh*fs)=B/(Tcoh*fs*fs)
%                     R_kf=4.092e6/(2*1e-3*Freq_sample^2);
                     %simga0^2=N0B=b/fs=2*fcode/fs
                    R_kf=4.092e6/(Freq_sample)/(Freq_sample*T_kf);  %34db �߶�̬����
                    CN0LIN=10^(CN0/10);
%                     R_kf=1/(2*T_coh*CN0LIN);
                    COHNUM=Freq_sample*T_kf;
                    R_kf=1/(A(AcqSatNUM)^2*COHNUM*(2*pi)^2);
                    R_kf=1/COHNUM;
                    %T�ǰ�  1ms
%                     R_kf=1/(8*pi*pi*T_coh*CN0LIN*2);
                   
%                     R_kf=2*A(1)^2/(8*pi*pi*T_coh*CN0LIN);
%                     R_kf=8e-6;
%                     R_kf=1/(8*pi*pi*T_coh*CN0LIN);
                    % ״̬��ʼ�� x = [��e; f; f?; f?]���ӵ�ǰ��·ֵ����
                    x_kf = zeros(4,1);
%                     x_kf(1) = mod(LocalDop*T_kf,1);            % ������λ��ֵ
                    x_kf(2) = LocalDop;     % ��ǰ������Ƶ�ʹ���
                    % f?, f? ��ʼ�� 0
                    P_kf = diag([5, 20^2, (20/2)^2, 10]);% P_kf = diag([pi^2, 100^2, 50^2, 10^2]);
                    AMP=(T_coh*Freq_sample);
                    kf_initialized = true;                
                   
                end
                
                % ���� �������µ���λ��������� pd ����
                P = corrcohsum_carr(mid);
%                 P=P/AMP;
                pd = atan(imag(P)/real(P)) / (2*pi);  %��λ����� ������
                
                
                % ���� 1. Ԥ�� ����
                x_pred = Phi_kf * x_kf+B_kf*[deltaagnel;f_nco];
                P_pred = Phi_kf * P_kf * Phi_kf' + Q_kf;                
                %����۲�Ĳв� inv���֣�Ҳ����zk-hk*xpred-ck*uk  ��λ��������
                rawinv=H_kf*x_pred-T_kf/2*(f_nco);
                innov=pd-rawinv;           
                PD=[PD pd];
                % ���� 2. ���� ����
                K_kf = P_pred * H_kf' / (H_kf * P_pred * H_kf' + R_kf);               
                x_kf = x_pred + K_kf * innov;
                P_kf = (eye(4) - K_kf * H_kf) * P_pred;
                f_nco = x_kf(2) + x_kf(3)*T_kf + x_kf(4)*T_kf^2/2;                
                carrier_dco_deltaL = (Fc + f_nco) / Freq_sample;
                deltaagnel= x_kf(1);
                carrier_dco_phaseL = carrier_dco_phaseL+deltaagnel;%��λ��������ֱ�Ӽ��ϱ����˲�֮�����λ���������룡
                % ��������״̬�����ƽ�              
                corrcohsum_carr = zeros(1,LengthCorreArray);
                FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
             
                fid = fopen(filename,'a+');
                fprintf(fid,'loopcnt %06d carr %+010.2f  pd%.3f  innov%.3f\n',...
                    loopcnt,carrier_dco_deltaL*Freq_sample-Fc,pd,innov);
                fclose(fid);
%                 fprintf('P_k�Խ���: [%.2e, %.2e, %.2e, %.2e]\n', diag(P_kf));
                
                
            end
        elseif(mode==3)    
              if(mod(mstime,2)==1)
                %% �������� ������mode==3 EKF�˲���֧ ��������
                
                % ���� ��һ�ν���ʱ��һ�γ�ʼ�� ����
                if ~exist('Ekf_initialized','var') || ~Ekf_initialized
                    % ʱ�䲽����ͬ��·�����ʣ�
                    T_kf = 2*1e-3;
                    %xk=��*xk-1+B*G
                    % ״̬ת�ƾ��� �� (4��4)
                    Phi_kf = [1, T_kf, T_kf^2/2, T_kf^3/6;
                        0,     1,      T_kf,    T_kf^2/2;
                        0,     0,        1,         T_kf;
                        0,     0,        0,           1];
                    %
                    B_kf=[-1 ,-T_kf;
                           0,0 ;
                           0,0;
                           0,0;];
                    %G��ӿ����� ��deltaagnel��λ�������==Xk-1(1),�Լ�f_nco��k-1��
                    f_nco=LocalDop;%f_nco��λ��Ƶ��
                    deltaagnel=0;
                    % ��������ǿ�� ��?�����Ը������Ӽ��ٶ� A_max ������                  
                    A_max = 500;
                    sigma_a = (A_max) * T_kf / 2;
                    Q_kf = sigma_a * [T_kf^6/252, T_kf^5/72, T_kf^4/30, T_kf^3/24;
                        T_kf^5/72, T_kf^4/20,  T_kf^3/8, T_kf^2/6;
                        T_kf^4/30,  T_kf^3/8,    T_kf^2/3,    T_kf/2;
                        T_kf^3/24,  T_kf^2/6,    T_kf/2,       1];
                   
                    % �۲���� H  ��KF��ͬ���۲����ÿһ�ζ��ǻ�ı�Ĺ۲ⷽ�̶Ը���״̬������ƫ���� 
                    H_kf = zeros(2,4);
                    HA=A(AcqSatNUM)*T_kf*Freq_sample;% AT
                    %H11=AT*sin(2pi��e k|k-1 )*(-2pi)   H21=AT*cos(2pi��e k|k-1 )*(+2pi)  
                    % �������� R                
                    %R_kf=simga0^2/Ncoh=simga0^2/(Tcoh*fs)=B/(Tcoh*fs*fs)
                    rtemp=4.092e6/(2*1e-3*Freq_sample^2);
                    R_kf=[ rtemp,0;0, rtemp];
                    % ״̬��ʼ�� x = [��e; f; f?; f?]���ӵ�ǰ��·ֵ����
                    x_kf = zeros(4,1);
%                     x_kf(1) = mod(LocalDop*T_kf,1);            % ������λ��ֵ
                    x_kf(2) = LocalDop;     % ��ǰ������Ƶ�ʹ���
                    % f?, f? ��ʼ�� 0
                    P_kf = diag([5, 20^2, (20/2)^2, 10]);% P_kf = diag([pi^2, 100^2, 50^2, 10^2]);
               
                    Ekf_initialized = true;                
                   
                end
                
                % ���� �������µ���λ��������� pd ����
                P = corrcohsum_carr(mid);
                %���ĵ�Ӱ����ô�� ���ܼ򵥵�ȡ����ֵ    innov=Z_kf-H_forinnov;  һ����Ӱ�� 
                %���룿����
                I_k=2*real(P)/HA;
                Q_k=2*imag(P)/HA;
                
                
                % ���� 1. Ԥ�� ����
                x_pred = Phi_kf * x_kf+B_kf*[deltaagnel;f_nco];
                P_pred = Phi_kf * P_kf * Phi_kf' + Q_kf;                
                %����۲�Ĳв� inv���֣�Ҳ����zk-hk*xpred-ck*uk  ��λ��������
                Z_kf=[I_k;Q_k];
                %H11=AT*sin(2pi��e k|k-1 )*(-2pi)
                H_kf(1,1)=sin(2*pi*x_pred(1))*(-2*pi);
                H_kf(2,1)=cos(2*pi*x_pred(1))*(2*pi); 
                
                H_kf(1,2)=sin(2*pi*x_pred(2))*(-pi)*T_kf;
                H_kf(2,2)=cos(2*pi*x_pred(2))*(pi)*T_kf;
                
                H_kf(1,3)=sin(2*pi*x_pred(3))*(-pi)*T_kf*T_kf/3;
                H_kf(2,3)=cos(2*pi*x_pred(3))*(pi)*T_kf*T_kf/3;
                
                H_kf(1,4)=sin(2*pi*x_pred(4))*(-pi)*T_kf*T_kf*T_kf/12;
                H_kf(2,4)=cos(2*pi*x_pred(4))*(pi)*T_kf*T_kf*T_kf/12;                
                %У��
                H_forinnov=[  cos(2 * pi * x_pred(1)); sin(2 * pi * x_pred(1))];
                innov=Z_kf-H_forinnov;
                
                % ���� 2. ���� ����
                K_kf = P_pred * H_kf' / (H_kf * P_pred * H_kf' + R_kf);
                x_kf = x_pred + K_kf * innov;
                P_kf = (eye(4) - K_kf * H_kf) * P_pred;
                f_nco = x_kf(2) + x_kf(3)*T_kf + x_kf(4)*T_kf^2/2;                
                carrier_dco_deltaL = (Fc + f_nco) / Freq_sample;
                deltaagnel= x_kf(1);
                carrier_dco_phaseL = carrier_dco_phaseL+deltaagnel;%��λ��������ֱ�Ӽ��ϱ����˲�֮�����λ���������룡
                % ��������״̬�����ƽ�              
                corrcohsum_carr = zeros(1,LengthCorreArray);
                FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
             
                fid = fopen(filename,'a+');
                fprintf(fid,'loopcnt %06d carr %+010.2f  pd%.3f\n',...
                    loopcnt,carrier_dco_deltaL*Freq_sample-Fc,pd);
                fclose(fid);
%                 fprintf('P_k�Խ���: [%.2e, %.2e, %.2e, %.2e]\n', diag(P_kf));                
                
             end              
        elseif(mode==4)%���������  ��ʱ����
            if(mod(mstime,2)==1)
                %% �������� ������mode==4 EKF�˲���֧������������ӵ��ĵ�Ԥ�� ��������
                
                % ���� ��һ�ν���ʱ��һ�γ�ʼ�� ����
                if ~exist('Ekf_initializedbit','var') || ~Ekf_initializedbit
                    % ʱ�䲽����ͬ��·�����ʣ�
                    T_kf = 2*1e-3;
                    %xk=��*xk-1+B*G
                    % ״̬ת�ƾ��� �� (4��4)
                    Phi_kf = [1, T_kf, T_kf^2/2, T_kf^3/6,0;
                        0,     1,      T_kf,    T_kf^2/2,0;
                        0,     0,        1,         T_kf,0;
                        0,     0,        0,           1,0;
                        0,0,0,0,0.95];
                    %
                    B_kf=[-1 ,-T_kf;
                        0,0 ;
                        0,0;
                        0,0;
                        0,0;];
                    %G��ӿ����� ��deltaagnel��λ�������==Xk-1(1),�Լ�f_nco��k-1��
                    f_nco=LocalDop;%f_nco��λ��Ƶ��
                    deltaagnel=0;
                    % ��������ǿ�� ��?�����Ը������Ӽ��ٶ� A_max ������
                    A_max = 1000;
                    sigma_a = (A_max) * T_kf / 2;
                    Q_kf = sigma_a * [T_kf^6/252, T_kf^5/72, T_kf^4/30, T_kf^3/24,0;
                        T_kf^5/72, T_kf^4/20,  T_kf^3/8, T_kf^2/6,0;
                        T_kf^4/30,  T_kf^3/8,    T_kf^2/3,    T_kf/2,0;
                        T_kf^3/24,  T_kf^2/6,    T_kf/2,       1,0;
                          0 ,0 ,0 ,0 ,1;                               ];
                    
                    % �۲���� H  ��KF��ͬ���۲����ÿһ�ζ��ǻ�ı�Ĺ۲ⷽ�̶Ը���״̬������ƫ����
                    H_kf = zeros(2,5);
                    HA=A(AcqSatNUM)*T_kf*Freq_sample;% AT
                    %H11=AT*sin(2pi��e k|k-1 )*(-2pi)   H21=AT*cos(2pi��e k|k-1 )*(+2pi)
                    % �������� R
                    %R_kf=simga0^2/Ncoh=simga0^2/(Tcoh*fs)=B/(Tcoh*fs*fs)
                    rtemp=4.092e6/(2*1e-3*Freq_sample^2);
                    R_kf=[ rtemp,0;0, rtemp];
                    % ״̬��ʼ�� x = [��e; f; f?; f?,Dk]���ӵ�ǰ��·ֵ����
                    x_kf = zeros(5,1);
                    %                     x_kf(1) = mod(LocalDop*T_kf,1);            % ������λ��ֵ
                    x_kf(2) = LocalDop;     % ��ǰ������Ƶ�ʹ���
                    % f?, f? ��ʼ�� 0
                    x_kf(5)=1;
                    P_kf = diag([5, 20^2, (20/2)^2, 10,2]);% P_kf = diag([pi^2, 100^2, 50^2, 10^2]);
                    predictbit=[];%���Ԥ��ı��ؿ����Բ��ԣ�
                    Ekf_initializedbit = true;
                    
                end
                
                % ���� �������µ���λ��������� pd ����
                P = corrcohsum_carr(mid);
                %���ĵ�Ӱ����ô�� ���ܼ򵥵�ȡ����ֵ    innov=Z_kf-H_forinnov;  һ����Ӱ��
                %���룿����
                I_k=2*real(P)/HA;
                Q_k=2*imag(P)/HA;
                
                
                % ���� 1. Ԥ�� ����
                x_pred = Phi_kf * x_kf+B_kf*[deltaagnel;f_nco];
                P_pred = Phi_kf * P_kf * Phi_kf' + Q_kf;
                %����۲�Ĳв� inv���֣�Ҳ����zk-hk*xpred-ck*uk  ��λ��������
                Z_kf=[I_k;Q_k];
                %H11=AT*sin(2pi��e k|k-1 )*(-2pi)
                H_kf(1,1)=sin(2*pi*x_pred(1))*(-2*pi)*x_pred(5);
                H_kf(2,1)=cos(2*pi*x_pred(1))*(2*pi)*x_pred(5);
                
%                 H_kf(1,2)=sin(2*pi*x_pred(2))*(-pi)*T_kf;
%                 H_kf(2,2)=cos(2*pi*x_pred(2))*(pi)*T_kf;
%                 
%                 H_kf(1,3)=sin(2*pi*x_pred(3))*(-pi)*T_kf*T_kf/3;
%                 H_kf(2,3)=cos(2*pi*x_pred(3))*(pi)*T_kf*T_kf/3;
%                 
%                 H_kf(1,4)=sin(2*pi*x_pred(4))*(-pi)*T_kf*T_kf*T_kf/12;
%                 H_kf(2,4)=cos(2*pi*x_pred(4))*(pi)*T_kf*T_kf*T_kf/12;
                
                H_kf(1,5)=cos(2*pi*x_pred(5));
                H_kf(2,5)=sin(2*pi*x_pred(5));
                
                %У��
                H_forinnov=[  x_pred(5)*cos(2 * pi * x_pred(1)); x_pred(5)*sin(2 * pi * x_pred(1))];
                innov=Z_kf-H_forinnov;
                
                % ���� 2. ���� ����
                K_kf = P_pred * H_kf' / (H_kf * P_pred * H_kf' + R_kf);
                x_kf = x_pred + K_kf * innov;
                x_kf(5)=sign(x_kf(5));
                predictbit=[predictbit x_kf(5)];
                P_kf = (eye(5) - K_kf * H_kf) * P_pred;
                f_nco = x_kf(2) + x_kf(3)*T_kf + x_kf(4)*T_kf^2/2;
                carrier_dco_deltaL = (Fc + f_nco) / Freq_sample;
                deltaagnel= x_kf(1);
                carrier_dco_phaseL = carrier_dco_phaseL+deltaagnel;%��λ��������ֱ�Ӽ��ϱ����˲�֮�����λ���������룡
                % ��������״̬�����ƽ�
                corrcohsum_carr = zeros(1,LengthCorreArray);
                FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
                
                fid = fopen(filename,'a+');
                fprintf(fid,'loopcnt %06d carr %+010.2f  pd%.3f\n',...
                    loopcnt,carrier_dco_deltaL*Freq_sample-Fc,pd);
                fclose(fid);
                %                 fprintf('P_k�Խ���: [%.2e, %.2e, %.2e, %.2e]\n', diag(P_kf));
                
            end
            
        elseif(mode==5)%CSK�����������
            
        elseif(mode==6)
            if(mod(mstime,2)==1)
                %% �������� ������mode==6 UKF�˲���֧������� ��������
                
                % ���� ��һ�ν���ʱ��һ�γ�ʼ�� ����
                if ~exist('Ukf_initializedbit','var') || ~Ukf_initializedbit
                    % ʱ�䲽����ͬ��·�����ʣ�
                    T_kf = 2*1e-3;
                    % ���� UKF ���� ���� 
                    nukf = 4;                        % ״̬ά��
                    alpha = 1e-3;  kappa = 0;  beta = 2;
                    lambda = alpha^2*(nukf+kappa)-nukf;
                    wm = [lambda/(nukf+lambda) repmat(1/(2*(nukf+lambda)),1,2*nukf)];
                    wc = wm;
                    wc(1) = wm(1)+(1-alpha^2+beta);                    
                    
                    Phi_kf = [1, T_kf, T_kf^2/2, T_kf^3/6;
                        0,     1,      T_kf,    T_kf^2/2;
                        0,     0,        1,         T_kf;
                        0,     0,        0,           1];
                    %
                    B_kf=[-1 ,-T_kf;
                           0,0 ;
                           0,0;
                           0,0;];
                    %G��ӿ����� ��deltaagnel��λ�������==Xk-1(1),�Լ�f_nco��k-1��
                    f_nco=LocalDop;%f_nco��λ��Ƶ��
                    
                    % ��������ǿ�� ��?�����Ը������Ӽ��ٶ� A_max ������
                    A_max = 500;
                    sigma_a = (A_max) * T_kf / 2;
                    Q_kf = sigma_a * [T_kf^6/252, T_kf^5/72, T_kf^4/30, T_kf^3/24;
                        T_kf^5/72, T_kf^4/20,  T_kf^3/8, T_kf^2/6;
                        T_kf^4/30,  T_kf^3/8,    T_kf^2/3,    T_kf/2;
                        T_kf^3/24,  T_kf^2/6,    T_kf/2,       1;  ];
                    
                    % �۲���� H  ��KF��ͬ���۲����ÿһ�ζ��ǻ�ı�Ĺ۲ⷽ�̶Ը���״̬������ƫ����
                    H_kf = zeros(2,4);
                    HA=A(AcqSatNUM)*T_kf*Freq_sample;% AT
                    deltaagnel=0;
                    % �������� R
                    %R_kf=simga0^2/Ncoh=simga0^2/(Tcoh*fs)=B/(Tcoh*fs*fs)
                    rtemp=4.092e6/(2*1e-3*Freq_sample^2);
                    R_kf=[ rtemp,0;0, rtemp];
                    % ״̬��ʼ�� x = [��e; f; f?; f?,Dk]���ӵ�ǰ��·ֵ����
                    x_kf = zeros(4,1);
                    %                     x_kf(1) = mod(LocalDop*T_kf,1);            % ������λ��ֵ
                    x_kf(2) = LocalDop;     % ��ǰ������Ƶ�ʹ���
                    % f?, f? ��ʼ�� 0
                   
                    P_kf = diag([5, 20^2, (20/2)^2, 10]);% P_kf = diag([pi^2, 100^2, 50^2, 10^2]);  
                    
                    Ukf_initializedbit = true;
                    % ���� ��ʼ״̬��Э���� ���� 
                    x_ukf = x_kf;                 % ��֮ǰ��֧�̳л��Զ���
                    P_ukf = P_kf;
                    Q_ukf = Q_kf;  R_ukf = R_kf;  % I/Q ����Э����                    
                end  
                % ���� ���� 2n+1 sigma ����� ���� �൱�� X ��K-1�� ��k-1���Ź���ѡȡ 2n+1 ������sigma
                S = chol((nukf+lambda)*P_ukf,'lower');
                sigmaukf = [ x_ukf, x_ukf+S, x_ukf-S ];  % ��С n��(2n+1)
                
                % ���� UKF Ԥ�� ����  �൱�� XK|k-1  
                % ����״̬ת�ƾ���2n+1������sigma����Ԥ�⣬Ԥ�⵽��k��ʱ�̵� 2n+1 ��Ԥ��sigma�� 
                for i=1:2*nukf+1
                    xi = sigmaukf(:,i);
                    % ״̬ת�ƣ�ͬ LKF �� Phi*xi + B*u
                    sigma_pred(:,i) = Phi_kf*xi + B_kf*[deltaagnel;f_nco];
                end
                %Ԥ��sigma������ֵ��Э����
                x_pred = sigma_pred * wm';              % ��ֵԤ��
                P_pred = Q_ukf;
                for i=1:2*nukf+1
                    d = sigma_pred(:,i)-x_pred;
                    P_pred = P_pred + wc(i)*(d*d');
                end                
                
                % ���� UKF �۲�Ԥ�� ����
                %Ԥ�� 2n+1 sigma�����۲ⷽ���У��õ�Ԥ��Ĺ۲�ֵ 
                for i=1:2*nukf+1
                    th = sigma_pred(1,i);
                    f  = sigma_pred(2,i);
                    % �����Թ۲� h(x) h��EKF��һ���ģ�����I Q�۲�ֵ��ע�����
                    z_sigma(:,i) = [cos(2*pi*th); sin(2*pi*th)];
                end
                z_pred = z_sigma * wm';                 % �۲��ֵ
                S_ukf = R_ukf;
                Pxz = zeros(nukf,2);
                for i=1:2*nukf+1
                    dz = z_sigma(:,i)-z_pred;
                    dx = sigma_pred(:,i)-x_pred;
                    S_ukf  = S_ukf + wc(i)*(dz*dz');
                    Pxz    = Pxz   + wc(i)*(dx*dz');
                end
                
                % ���� UKF ���� ����
                K_ukf = Pxz / S_ukf;
                P_ukf = P_pred - K_ukf*S_ukf*K_ukf';
                % ��ȡ��ʵ���� I/Q
                %ע����ģ���
                Pxy = corrcohsum_carr(mid);
                z    = [2*real(Pxy)/HA; 2*imag(Pxy)/HA];
                innov = z - z_pred;
                x_ukf = x_pred + K_ukf * innov;                
                
                                
                f_nco = x_ukf(2) + x_ukf(3)*T_kf + x_ukf(4)*T_kf^2/2;
                carrier_dco_deltaL = (Fc + f_nco) / Freq_sample;
                deltaagnel= x_ukf(1);
                carrier_dco_phaseL = carrier_dco_phaseL+deltaagnel;%��λ��������ֱ�Ӽ��ϱ����˲�֮�����λ���������룡
                % ��������״̬�����ƽ�
                corrcohsum_carr = zeros(1,LengthCorreArray);
                FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
                
                fid = fopen(filename,'a+');
                fprintf(fid,'loopcnt %06d carr %+010.2f  pd%.3f\n',...
                    loopcnt,carrier_dco_deltaL*Freq_sample-Fc,pd);
                fclose(fid);
                %                 fprintf('P_k�Խ���: [%.2e, %.2e, %.2e, %.2e]\n', diag(P_kf));
                
            end
            
            
        elseif(mode==7)%Ԥ��
            
        elseif(mode==8)%LKF 1ms Ԥ�⣬2ms����
            
            %% �������� ������mode==8 �������˲���֧ �������� 1msԤ�⣬2ms����
            
            % ���� ��һ�ν���ʱ��һ�γ�ʼ�� ����
            if ~exist('kf_initialized','var') || ~kf_initialized
                % ʱ�䲽����ͬ��·�����ʣ�
                T_kf = 1*1e-3;                
                %xk=��*xk-1+B*G
                % ״̬ת�ƾ��� �� (4��4)
                Phi_kf = [1, T_kf, T_kf^2/2, T_kf^3/6;
                    0,     1,      T_kf,    T_kf^2/2;
                    0,     0,        1,         T_kf;
                    0,     0,        0,           1];
                %
                B_kf=[-1 ,-T_kf;
                    0,0 ;
                    0,0;
                    0,0;];
                %G��ӿ����� ��deltaagnel��λ�������==Xk-1(1),�Լ�f_nco��k-1��
                f_nco=LocalDop;%f_nco��λ��Ƶ��
                deltaagnel=0;
                % ��������ǿ�� ��?�����Ը������Ӽ��ٶ� A_max ������
                A_max = 10;
                sigma_a = (A_max) * T_kf / 2;
                Q_kf = sigma_a * [T_kf^6/252, T_kf^5/72, T_kf^4/30, T_kf^3/24;
                    T_kf^5/72, T_kf^4/20,  T_kf^3/8, T_kf^2/6;
                    T_kf^4/30,  T_kf^3/8,    T_kf^2/3,    T_kf/2;
                    T_kf^3/24,  T_kf^2/6,    T_kf/2,       1];
                
                % �۲���� H
                H_kf = [1, T_kf*1/2, T_kf^2/6, T_kf^3/24];
                
                % �������� R
                %R_kf=simga0^2/Ncoh=simga0^2/(Tcoh*fs)=B/(Tcoh*fs*fs)
                %                     R_kf=4.092e6/(2*1e-3*Freq_sample^2);
                %simga0^2=N0B=b/fs=2*fcode/fs
%                 R_kf=4.092e6/(Freq_sample)/(Freq_sample*T_kf);
%               R_kf=1/(8pi^2*CNO*Tcoh)
                CN0LIN=10^(CN0/10);
                R_kf=1/(8*pi*pi*T_kf*CN0LIN);
                
                % ״̬��ʼ�� x = [��e; f; f?; f?]���ӵ�ǰ��·ֵ����
                x_kf = zeros(4,1);
                %                     x_kf(1) = mod(LocalDop*T_kf,1);            % ������λ��ֵ
                x_kf(2) = LocalDop;     % ��ǰ������Ƶ�ʹ���
                % f?, f? ��ʼ�� 0
                P_kf = diag([5, 20^2, (20/2)^2, 10]);% P_kf = diag([pi^2, 100^2, 50^2, 10^2]);
                
                kf_initialized = true;
                
            end
            
            % ���� �������µ���λ��������� pd ����
            P = corrcohsum_carr(mid);
            pd = atan(imag(P)/real(P)) / (2*pi);  %��λ����� ������         
          
            
            
            % ���� 1. Ԥ�� ����
            x_pred = Phi_kf * x_kf+B_kf*[deltaagnel;f_nco];
            P_pred = Phi_kf * P_kf * Phi_kf' + Q_kf;
            %����۲�Ĳв� inv���֣�Ҳ����zk-hk*xpred-ck*uk  ��λ��������
            if(mod(mstime,2)==1)
                PD=[PD pd];
                rawinv=H_kf*x_pred-T_kf/2*(f_nco);
                innov=pd-rawinv;
                K_kf = P_pred * H_kf' / (H_kf * P_pred * H_kf' + R_kf);
                x_kf = x_pred + K_kf * innov;
                P_kf = (eye(4) - K_kf * H_kf) * P_pred;
            end
            
            % ���� 2. ���� ����
            
            f_nco = x_kf(2) + x_kf(3)*T_kf + x_kf(4)*T_kf^2/2;
            carrier_dco_deltaL = (Fc + f_nco) / Freq_sample;
            deltaagnel= x_kf(1);
            carrier_dco_phaseL = carrier_dco_phaseL+deltaagnel;%��λ��������ֱ�Ӽ��ϱ����˲�֮�����λ���������룡
            % ��������״̬�����ƽ�
            corrcohsum_carr = zeros(1,LengthCorreArray);
            FCARR = [FCARR carrier_dco_deltaL*Freq_sample-Fc];
            
            fid = fopen(filename,'a+');
            fprintf(fid,'loopcnt %06d carr %+010.2f  pd%.3f  innov%.3f\n',...
                loopcnt,carrier_dco_deltaL*Freq_sample-Fc,pd,innov);
            fclose(fid);
            %                 fprintf('P_k�Խ���: [%.2e, %.2e, %.2e, %.2e]\n', diag(P_kf));
            
            
            
            
        end       
        %���㱾������ܶ����ղ�ֵ
        freceive=Freq_IF(AcqSatNUM)-Fc+freq_dot(AcqSatNUM)*mstime*0.001+1/2*(mstime*0.001)^2*freq_dot_2rd(AcqSatNUM);
        doperror=freceive-carrier_dco_deltaL*Freq_sample+Fc;
        DOPERRORre=[DOPERRORre doperror];
        if(abs(doperror)>1000)%dop������1000Hz ��������
            break;
        end
        if(mode~=5)
        %�뻷·
            if( mod(mscountL1_bitperiod,codeloop_cohtime) == 0)
                codeloop_noncohcnt = codeloop_noncohcnt + 1;
                corrnoncohsum_code = corrnoncohsum_code + abs(corrcohsum_code).^2;
                corrcohsum_code =  zeros(1,LengthCorreArray);
                if(codeloop_noncohcnt == codeloop_noncohtime)
                    codeloop_noncohcnt = 0;
                    E = corrnoncohsum_code(mid - 5);
                    L = corrnoncohsum_code(mid + 5);
                    cd = (E-L)/(E+L);
                    vo = vo_1 + cd1*(cd - cd_1) + cd2*(cd + cd_1);  %VCO����
                    code_dco_deltaL = code_dco_deltaL - (vo - vo_1)/Freq_sample; %���ظ�����Ƶ�ʸ���
    %                 code_dco_deltaL*Freq_sample-Freq_code
                    FCODE = [FCODE;loopcnt code_dco_deltaL*Freq_sample-Freq_code];              %���ظ�����Ƶ�ʴ�������

                    vo_1 = vo;
                    cd_1 = cd;
                    corrnoncohsum_code =  zeros(1,LengthCorreArray);
                end
            end
            jugujitime=80;
            if( mod(mstime,jugujitime) == 0)
                  %������ȹ��Ʋ����ã�     

    % %             if(cntL >= 2)
    %                 tmmp = accum;
    %                 NBP = abs(sum(tmmp))^2;
    %                 I2_Q2 = real(sum(tmmp))^2-imag(sum(tmmp))^2;
    %                 WBP = tmmp * tmmp';
    %                 NP = NBP/WBP;
    %                 loopcnt;
    %                 tt1=10*log10((NP-1)/(20-NP)*1000);
    %                 pld = I2_Q2/NBP;
    % %                 carrier_dco_deltaL*Freq_sample-Fc
    % %                 code_dco_deltaL*Freq_sample-Freq_code
    %                 NParray = [NParray NP];
    %                 PLD = [PLD pld];
    %                 PD = [PD pd];
    %                 accum = [];
    %                 CN0L=[CN0L tt1];
    %                 cntL = 0;
    %                 
    %                 fid = fopen(filename,'a');
    %                 fprintf(fid,'loopcnt %06d carr %+010.2f code %+010.2f NP %+010.2f PLD %+010.2f\n',...
    %                     loopcnt,carrier_dco_deltaL*Freq_sample-Fc,code_dco_deltaL*Freq_sample-Freq_code,NP,pld);
    %                 fclose(fid);
    % %             end

               %�ĳɾع���
               Tju=jugujitime/2;
               M2CO=M2/Tju;
               M4CO=M4/Tju;
               tt1=10*log10((sqrt(2*M2CO^2-M4CO))/(M2CO-sqrt(2*M2CO^2-M4CO))*1000);
               CN0L=[CN0L tt1];
               M2=0;
               M4=0;

            end
    %         if(loopcnt >= 100000)
    %             
    %             figure,plot(abs(Parr),'r'),hold on,plot(abs(Earr),'k'),plot(abs(Larr),'b')
    %             figure,plot(FCODE(:,1),FCODE(:,2)),title('freqcode')
    %             figure,plot(FCARR),title('freqcarrier')
    %             figure,plot(abs(CN0L)),title('����ȹ���')
    % %             figure,plot(PD),title('PD')
    %             figure,plot(PDFLL),title('PDFLL')
    %             figure,plot(DOPERRORre),title('��������ܶ����ղ�ֵ')
    %             figure,plot(carrerr),title('carrierphaseerr')
    %             figure,plot(codeerr),title('codephaseerr')
    % %             figure,plot(NParray),title('NP')
    % %             NP=mean(NParray(50:end)),tt1=10*log10((NP-1)/(20-NP)*1000);
    %             break;
    %         end
        end        
    end
    %%
    %�ź�����ģ��
    result = randn*sigma + sqrt(-1)*randn*sigma; %%%%%%%% ����IQ��·�ź�����
    %%%%%%%% ����Ƶ������Ƶ�任�õ����ս�� %���ٵ����Լ�ӳ����CA���
    for ss= 1:4
        result = result + A(ss)*codelow(ss,chipcounter(ss)+1)*(1-gt(ss))*BITlow(bitnumlow(ss)+1)*exp(sqrt(-1)*2*pi*carrier_dco_phase(ss))+...
                          A(ss)*codehigh(ss,chipcounter(ss)+1)*gt(ss)*exp(sqrt(-1)*2*pi*carrier_dco_phase(ss));       
      
    end
    tmptmp = floor(mod(chipcntL+code_dco_phaseL+LocalCorreArray+codelength,codelength)+1); %������Ƶ�����Ӧ����Ƭ����
    tmp1 = LocalCode(tmptmp);   %�ɱ�����Ƶ�����Ӧ����Ƭ�������õ���Ӧ��Ƭ
    
    csktmptmp=floor(mod(chipcntL+code_dco_phaseL+codelength,codelength)+1); 
    tmpcskcode=Localhightraverse(:,csktmptmp);
%     corrsum = corrsum + result*exp(-sqrt(-1)*2*pi*carrier_dco_phase(AcqSatNUM))*tmp1;
    if(mod(mscount(AcqSatNUM),2)==0)
        if(mode==3||mode==6) 
            corrsum = corrsum + result*exp(-sqrt(-1)*2*pi*carrier_dco_phaseL)*tmp1*BITlow(bitnumlow(AcqSatNUM)+1);%mode==3 EKF��֧��IQ�۲�ֵ��Ҫ�������
        else
            corrsum = corrsum + result*exp(-sqrt(-1)*2*pi*carrier_dco_phaseL)*tmp1;
        end
    else
        %��ɽ��
        corrsumforcskdemodulate=corrsumforcskdemodulate+result*exp(-sqrt(-1)*2*pi*carrier_dco_phaseL).*tmpcskcode';       
    end
    n = n + 1;
    for i = 1:4
        %%% ��λp = 2*pi*��f dt (1); Ƶ�ʶ�ʱ��Ļ���
        %%% f = f0 + ��f' dt    (2); �������Ƶ�ʼ��ٶ�
        %%% �õ�p = 2*pi*(f0*t + f'*t^2/2)����ɢ����
        %%% p(n) = f0 * nTs + f'* (n*Ts)^2 / 2
        %%% p(n-1) = f0*(n-1)Ts + f'(n-1)^2*Ts^2/2
        %%% ��p(n) - p(n-1)�õ����¹�ʽ�����мӼ��ٶȼ����߽ױ仯�ʣ����մ˹��̵����ں���Ӹ߽���
        carrier_dco_phase(i) = carrier_dco_phase(i) + carrier_dco_delta(i) + 0.5*freq_dot(i)*(2*n-1)/Freq_sample/Freq_sample+1/6*freq_dot_2rd(i)*(3*n^2-3*n+1)/Freq_sample^3; 
        code_dco_phase(i) = code_dco_phase(i)+ code_dco_delta(i) + 0.5*freq_code_dot(i)*(2*n-1)/Freq_sample/Freq_sample+1/6*freq_code_dot_2rd(i)*(3*n^2-3*n+1)/Freq_sample^3; 
        
        %%% ��α����λ����1��˵�������Ѿ���һ����Ƭ����ʱ��Ƭ��������һ
        if(carrier_dco_phase(i) > 1)                                                                               
            carrier_dco_phase(i) = carrier_dco_phase(i) - 1;                                                           
        end                                                                                                                   
        if(code_dco_phase(i) > 1)
            code_dco_phase(i) = code_dco_phase(i) - 1;                          
            chipcounter(i) = chipcounter(i) + 1; 
            %%% α�볤��codelength����Ƭ�����������Ƶ����򾭹�һ��α�����ڣ����������㣬��һ����
            if(chipcounter(i) >= codelength)
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
    
    
    %%%%%%%%���ظ����źŵĵ�������
    localn = localn + 1;
    carrier_dco_phaseL = carrier_dco_phaseL + carrier_dco_deltaL ;
    if(carrier_dco_phaseL > 1)
        carrier_dco_phaseL = carrier_dco_phaseL - 1;
    end
    code_dco_phaseL = code_dco_phaseL + code_dco_deltaL; 
   

%     if(localn == round(Freq_sample * 0.001))%ÿ1ms����һ�ζ����ղ���������0.001��ʾ1ms
%         localn = 0;
%         carrier_dco_deltaL = carrier_dco_deltaL + (freq_dot(AcqSatNUM)*0.001)/Freq_sample;
%         %��Ƶ������ز�����Ҫ��������Ϊ�仯��̫��
%     end
    if(code_dco_phaseL > 1)
        code_dco_phaseL = code_dco_phaseL - 1;
        chipcntL = chipcntL + 1;
        if(chipcntL >= codelength)
            dump = 1;
            chipcntL = 0;
            mscountL = mscountL + 1;
            mstime=mstime+1;
                %%% ��������50bps�����������Ƶ����򾭹�һ�����ڣ���Ӧһ�������룬������������㣬���ؼ�������һ
            if(mscountL >= bitperiod)
                mscountL = 0;
%                 bitnumL = bitnumL+1;
                
            end
        end
    end
    
     if(mod(localn,Freq_sample * 0.001) == 0)          %ÿ1ms����һ�¸������
%         localn = 0;
        cod = code_dco_phaseL - code_dco_phase(AcqSatNUM);
        car = carrier_dco_phaseL - carrier_dco_phase(AcqSatNUM);
        codeerr = [codeerr cod];
        carrerr = [carrerr car];
    end
    %%
    %��������ȣ��ı价·����
%     if(n == Freq_sample*3)
%         CN0 = 40;
%         A(1) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%50dBHz
%         A(2) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%45dBHz
%         A(3) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%40dBHz
%         A(4) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%37dBHz
%         codeloop_cohtime = 20;
%         codeloop_noncohtime = 10;
%         codeloop_noncohcnt = 0;
%         bn_code = 0.5;
%         cd1 = 8/3*bn_code;    %�����뻷����һ��K1����ʱ��Ч��������1.5Hz
%         cd2 = cd1*cd1/2*T*codeloop_cohtime*codeloop_noncohtime/2;   %�����뻷��������K2
% 
%         carrierloop_cohtime = 20;            %��·����ʱ�䣬��λms
%         bn_carr = 5;                       %��·����
%         cp1_2nd = 8/3*bn_carr;                  %���׻�·ϵ��
%         cp2_2nd = cp1_2nd*cp1_2nd/2*T*carrierloop_cohtime/2;
%         cp1_3rd = 60/23*bn_carr;                  %���׻�·ϵ��
%         cp2_3rd = cp1_3rd*cp1_3rd*4/9*T*carrierloop_cohtime/2;
%         cp3_3rd = (cp1_3rd*cp1_3rd*cp1_3rd)*2.0/27.0* (T*carrierloop_cohtime) * (T*carrierloop_cohtime)/4.0;
%     end
%     if(n == Freq_sample*6)
%         CN0 = 30;
%         A(1) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%50dBHz
%         A(2) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%45dBHz
%         A(3) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%40dBHz
%         A(4) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%37dBHz
%         codeloop_cohtime = 20;
%         codeloop_noncohtime = 10;
%         codeloop_noncohcnt = 0;
%         bn_code = 0.5;
%         cd1 = 8/3*bn_code;    %�����뻷����һ��K1����ʱ��Ч��������1.5Hz
%         cd2 = cd1*cd1/2*T*codeloop_cohtime*codeloop_noncohtime/2;   %�����뻷��������K2
% 
%         carrierloop_cohtime = 20;            %��·����ʱ�䣬��λms
%         bn_carr = 5;                       %��·����
%         cp1_2nd = 8/3*bn_carr;                  %���׻�·ϵ��
%         cp2_2nd = cp1_2nd*cp1_2nd/2*T*carrierloop_cohtime/2;
%         cp1_3rd = 60/23*bn_carr;                  %���׻�·ϵ��
%         cp2_3rd = cp1_3rd*cp1_3rd*4/9*T*carrierloop_cohtime/2;
%         cp3_3rd = (cp1_3rd*cp1_3rd*cp1_3rd)*2.0/27.0* (T*carrierloop_cohtime) * (T*carrierloop_cohtime)/4.0;
% 
%     end
%     
%     if(n == Freq_sample*9)
%         CN0 = 20;
%         A(1) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%50dBHz
%         A(2) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%45dBHz
%         A(3) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%40dBHz
%         A(4) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%37dBHz
%         codeloop_cohtime = 20;
%         codeloop_noncohtime = 10;
%         codeloop_noncohcnt = 0;
%         bn_code = 0.5;
%         cd1 = 8/3*bn_code;    %�����뻷����һ��K1����ʱ��Ч��������1.5Hz
%         cd2 = cd1*cd1/2*T*codeloop_cohtime*codeloop_noncohtime/2;   %�����뻷��������K2
% 
%         carrierloop_cohtime = 20;            %��·����ʱ�䣬��λms
%         bn_carr = 2;                       %��·����
%         cp1_2nd = 8/3*bn_carr;                  %���׻�·ϵ��
%         cp2_2nd = cp1_2nd*cp1_2nd/2*T*carrierloop_cohtime/2;
%         cp1_3rd = 60/23*bn_carr;                  %���׻�·ϵ��
%         cp2_3rd = cp1_3rd*cp1_3rd*4/9*T*carrierloop_cohtime/2;
%         cp3_3rd = (cp1_3rd*cp1_3rd*cp1_3rd)*2.0/27.0* (T*carrierloop_cohtime) * (T*carrierloop_cohtime)/4.0;
% 
%     end
%     
%     if(n == Freq_sample*90)
%         CN0 = 40;
%         A(1) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%50dBHz
%         A(2) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%45dBHz
%         A(3) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%40dBHz
%         A(4) = sigma* sqrt(2*10^(CN0/10)/Freq_sample);%37dBHz
%         codeloop_cohtime = 20;
%         codeloop_noncohtime = 10;
%         codeloop_noncohcnt = 0;
%         bn_code = 0.5;
%         cd1 = 8/3*bn_code;    %�����뻷����һ��K1����ʱ��Ч��������1.5Hz
%         cd2 = cd1*cd1/2*T*codeloop_cohtime*codeloop_noncohtime/2;   %�����뻷��������K2
% 
%         carrierloop_cohtime = 20;            %��·����ʱ�䣬��λms
%         bn_carr = 2;                       %��·����
%         cp1_2nd = 8/3*bn_carr;                  %���׻�·ϵ��
%         cp2_2nd = cp1_2nd*cp1_2nd/2*T*carrierloop_cohtime/2;
%         cp1_3rd = 60/23*bn_carr;                  %���׻�·ϵ��
%         cp2_3rd = cp1_3rd*cp1_3rd*4/9*T*carrierloop_cohtime/2;
%         cp3_3rd = (cp1_3rd*cp1_3rd*cp1_3rd)*2.0/27.0* (T*carrierloop_cohtime) * (T*carrierloop_cohtime)/4.0;
% 
%     end
    if(n > Freq_sample*runtime)
%        figure, stem(histnumi);
%        figure, stem(histnumq);
        break
    end
    
    %%%%%%%% sec��һ�벢��ӡ
    samplenum = samplenum + 1;
    if(samplenum == Freq_sample)
        samplenum = 0;
        sec = sec + 1;
        fprintf('%ds....\n',sec);
    end
end
%%
%ͳ��CSK������BER
errsumnoco=0;
errsumco=0;
time=runtime;
for index=1:1:(time*3000)
    errsumnoco=errsumnoco+abs(BIThigh(index)-highbitcsknoco(index));
    errsumco=errsumco+abs(BIThigh(index)-highbitcskco(index));
end
cskerrpercentnoco=errsumnoco/(time*3000);
cskerrpercentco=errsumco/(time*3000);
%ͳ��CSK������SER
Sersumno=0;
Sersumco=0;
for index=1:1:(time*500)
    if(highbitcsknoco_symbol(index)~=BITHIGHTRANS(index))
        Sersumno=Sersumno+1;
    end
    if(highbitcskco_symbol(index)~=BITHIGHTRANS(index))
        Sersumco=Sersumco+1;
    end
end
SERNO=Sersumno/(time*500);%FEI
SERCO=Sersumco/(time*500);%XIANGG
% fclose(fid);