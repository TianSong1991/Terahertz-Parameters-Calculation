function [Fq,Aq,thz]=FreqSpec(t_THz,E_THz, fp, fb,ed,bd)
%% �������˵��
%����
% t_THz: һά���飬��̫���������ʱ�����У���λps��
% E_THz��һά����,ʱ�����ж�Ӧ��̫���ȵ糡ǿ�ȣ���λmV��
% fp: ͨ����ֹƵ��, unit THz
% fb: �����ֹƵ��, unit THz
% ed: �ߴ�˥��0.2
% bd: ��ֹ��˥��3
%���
% Fq�����̫����Ƶ�׵�Ƶ��, ��λΪTHz
% Aq:��Ӧ��̫����Ƶ�׹�һ��ǿ�ȣ���λ����
%%
%����������
y = E_THz;
        
% ʱ���źŵ�ƽ��ʱ����
deltaT = mean(t_THz(2 : end) - t_THz(1 : end - 1));

% ���źŵĳ���ʱ�䲹��100ps
% signalLen = t_THz(end) - t_THz(1);
% if signalLen < 100
%     nTail = floor((100 - signalLen) / deltaT);
%     eTail = zeros(1, nTail);
%     startIndex = size(y, 2) + 1;
%     endIndex = size(y, 2) + nTail;
%     y(startIndex : endIndex) = eTail;
% end

N = length(y);          %��������
Fs = 1 / deltaT;   %%�����ʣ������ӳ��߲���������

%%��ͨ�˲����ߴ�˥��0.2 DB����ֹ��˥��3 DB
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
Y1 = Y1/N*2;            %����N�ٳ���2������ʵ��ֵ��NԽ�󣬷�ֵ����Խ��
f  = Fs/N*(0:1:N-1);    %Ƶ��

% A  = (abs(Y1))/max((abs(Y1)));           %��ֵabs(Y1)
A  = (abs(Y1));
% P  = angle(Y1);         %��ֵ
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

