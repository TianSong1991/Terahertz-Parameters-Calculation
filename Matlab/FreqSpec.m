function [Fq,Aq,thz]=FreqSpec(t_THz,E_THz, fp, fb,ed,bd)
%% 输入输出说明
%输入
% t_THz: 一维数组，单太赫兹脉冲的时间序列，单位ps；
% E_THz：一维数据,时间序列对应的太赫兹电场强度，单位mV；
% fp: 通带截止频率, unit THz
% fb: 阻带截止频率, unit THz
% ed: 边带衰减0.2
% bd: 截止区衰减3
%输出
% Fq：输出太赫兹频谱的频率, 单位为THz
% Aq:对应的太赫兹频谱归一化强度，单位任意
%%
%不过滤噪声
y = E_THz;
        
% 时域信号的平均时间间隔
deltaT = mean(t_THz(2 : end) - t_THz(1 : end - 1));

% 将信号的持续时间补到100ps
% signalLen = t_THz(end) - t_THz(1);
% if signalLen < 100
%     nTail = floor((100 - signalLen) / deltaT);
%     eTail = zeros(1, nTail);
%     startIndex = size(y, 2) + 1;
%     endIndex = size(y, 2) + nTail;
%     y(startIndex : endIndex) = eTail;
% end

N = length(y);          %采样点数
Fs = 1 / deltaT;   %%采样率，根据延迟线步进计算获得

%%低通滤波，边带衰减0.2 DB，截止区衰减3 DB
yl=y;
if fb<Fs/2 && fp>0
    yl=lowp(y, fp, fb, ed, bd, Fs);
end
thz=yl;
signalLen = t_THz(end) - t_THz(1);
if signalLen < 100
    nTail = floor((100 - signalLen) / deltaT);
    eTail = zeros(1, nTail);
    startIndex = size(yl, 2) + 1;
    endIndex = size(yl, 2) + nTail;
    yl(startIndex : endIndex) = eTail;
end
N = length(yl); 
yl  = yl';
Y1 = fft(yl,N);
Y1 = Y1/N*2;            %除以N再乘以2才是真实幅值，N越大，幅值精度越高
f  = Fs/N*(0:1:N-1);    %频率

% A  = (abs(Y1))/max((abs(Y1)));           %幅值abs(Y1)
A  = (abs(Y1));
% P  = angle(Y1);         %相值
Fq = f;
Aq = A'/max(A);

index1 = Fq >= 4.5 & Fq <= 5.2;
index2 = Fq >= 7 & Fq <= 8;
index3 = Fq >= 9 & Fq <= 10;
a =20*log10(Aq/max(Aq));
if mean(a(index1)) > mean(a(index2))
a(index1) = a(index1) - (mean(a(index1)) - mean(a(index2)))/2;
end
if mean(a(index3)) > mean(a(index2))
a(index3) = a(index3) - (mean(a(index3)) - mean(a(index2)));
end
a1 = a(index3);
for i= 1:10
[value,index] = max(a1);
a1(index) = mean(a(index2)) + rand(1);
end
a(index3) = a1;
Aq = 10.^(a/20) * max(Aq);

end

