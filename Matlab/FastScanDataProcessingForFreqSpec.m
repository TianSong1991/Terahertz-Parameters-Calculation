function [FastTHz_Time,Fast_THzSignal,ps]=FastScanDataProcessingForFreqSpec(FastScanData,nda,fp)
%用于快扫描太赫兹信号频域分析的数据解析
%输入
%FastScanData 读取快扫描的数据
%nda 单次总采样点数 fp;分频数
%输出
%FastTHz_Time:1行N列数组,为时间坐标;
%Fast_THzSignal,1行N列数组，为时间坐标对应的太赫兹信号;
%dat2_adj,M行N列数组，每一行代表一次扫描的太赫兹波形号；
% 数据读入，Data Importing
warning off all
marker_vec=[163 163 165 165];% 标记位矢量,Marker Vector to Determine
tn35=strfind(FastScanData,marker_vec);%求标记位位置，Position of marker vector
%nda=nda/fp;% 数据点数，Number of Sampling Points
%去除漏采的数据点，该方法虽然简洁但是无法打包，只能用笨办法
dist=diff(tn35);% 求两个相邻标记位之间的点数 Find the distance between neighbor position of marker vectors
pos_miss=find(dist~=2*nda+20);%如果点数小于2nda,表明出现漏采 Find the missing points if the distance between two neighbor position is smaller than 2*nda
for i = 1:length(pos_miss)
    if dist(pos_miss(i)) > 2*nda+20
        num = dist(pos_miss(i)) - 2*nda - 20;
        data = FastScanData(tn35(pos_miss(i))+4:tn35(pos_miss(i))+3+num);
        if all(data == 165)
            tn35(pos_miss(i)) = tn35(pos_miss(i)) + num;
            dist(pos_miss(i)) = dist(pos_miss(i)) - num;
        end
    end
end
pos_miss=find(dist~=2*nda+20);
if isempty(pos_miss)==0
    tn35(pos_miss)=[];%将漏采的数据去除
end
N_wave=length(tn35)-1;%有用的太赫兹信号数目
dat2=zeros(nda,N_wave);%原始太赫兹信号初始化
dat2_adj=zeros(nda,N_wave);%去背景后太赫兹信号的初始化
t=(0:1:nda-1)';
FastTHz_Time=(t*0.5*fp*6/300)';%太赫兹信号的时间坐标，单位ps
for i=1:N_wave
    ttda2=FastScanData(tn35(i)+4:tn35(i)+2*nda+3);
    dat2(:,i)=ttda2(1:2:end)+ttda2(2:2:end)*256;%太赫兹信号的解码
    OrgTHz_i=dat2(:,i);%原始太赫兹信号
    [para,~,mu]=polyfit(t,OrgTHz_i,10);%求解背景，利用十次多项式拟合
    Val_pf=polyval(para,t,[],mu);%拟合的背景值
    dat2_adj(:,i)=(OrgTHz_i-Val_pf);%去背景后的真实太赫兹信号%位置解码
end
dat2_adj=dat2_adj';
Fast_THzSignal=mean(dat2_adj);
unit_ps=floor(1/(0.5*fp*6/300));
 [~,locs] = findpeaks(Fast_THzSignal,'Npeaks',20,'MinPeakHeight',max(Fast_THzSignal)*0.2,'MinPeakDistance',floor(1*unit_ps));%第二反射峰与第一反射峰的最小时间间隔为2 ps（最小间隔2ps应该太大了吧，我觉得可以是0.5ps）
 ps=[];
 if length(locs)>1
     for i=2:length(locs)
            ps(end+1)=(locs(i)-locs(i-1))*0.5*fp*6/300';
     end
end
