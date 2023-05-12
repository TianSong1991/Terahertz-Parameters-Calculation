function [FastTHz_Time,Fast_THzSignal,ps]=FastScanDataProcessingForFreqSpec(FastScanData,nda,fp)
%���ڿ�ɨ��̫�����ź�Ƶ����������ݽ���
%����
%FastScanData ��ȡ��ɨ�������
%nda �����ܲ������� fp;��Ƶ��
%���
%FastTHz_Time:1��N������,Ϊʱ������;
%Fast_THzSignal,1��N�����飬Ϊʱ�������Ӧ��̫�����ź�;
%dat2_adj,M��N�����飬ÿһ�д���һ��ɨ���̫���Ȳ��κţ�
% ���ݶ��룬Data Importing
warning off all
marker_vec=[163 163 165 165];% ���λʸ��,Marker Vector to Determine
tn35=strfind(FastScanData,marker_vec);%����λλ�ã�Position of marker vector
%nda=nda/fp;% ���ݵ�����Number of Sampling Points
%ȥ��©�ɵ����ݵ㣬�÷�����Ȼ��൫���޷������ֻ���ñ��취
dist=diff(tn35);% ���������ڱ��λ֮��ĵ��� Find the distance between neighbor position of marker vectors
pos_miss=find(dist~=2*nda+20);%�������С��2nda,��������©�� Find the missing points if the distance between two neighbor position is smaller than 2*nda
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
    tn35(pos_miss)=[];%��©�ɵ�����ȥ��
end
N_wave=length(tn35)-1;%���õ�̫�����ź���Ŀ
dat2=zeros(nda,N_wave);%ԭʼ̫�����źų�ʼ��
dat2_adj=zeros(nda,N_wave);%ȥ������̫�����źŵĳ�ʼ��
t=(0:1:nda-1)';
FastTHz_Time=(t*0.5*fp*6/300)';%̫�����źŵ�ʱ�����꣬��λps
for i=1:N_wave
    ttda2=FastScanData(tn35(i)+4:tn35(i)+2*nda+3);
    dat2(:,i)=ttda2(1:2:end)+ttda2(2:2:end)*256;%̫�����źŵĽ���
    OrgTHz_i=dat2(:,i);%ԭʼ̫�����ź�
    [para,~,mu]=polyfit(t,OrgTHz_i,10);%��ⱳ��������ʮ�ζ���ʽ���
    Val_pf=polyval(para,t,[],mu);%��ϵı���ֵ
    dat2_adj(:,i)=(OrgTHz_i-Val_pf);%ȥ���������ʵ̫�����ź�%λ�ý���
end
dat2_adj=dat2_adj';
Fast_THzSignal=mean(dat2_adj);
unit_ps=floor(1/(0.5*fp*6/300));
 [~,locs] = findpeaks(Fast_THzSignal,'Npeaks',20,'MinPeakHeight',max(Fast_THzSignal)*0.2,'MinPeakDistance',floor(1*unit_ps));%�ڶ���������һ��������Сʱ����Ϊ2 ps����С���2psӦ��̫���˰ɣ��Ҿ��ÿ�����0.5ps��
 ps=[];
 if length(locs)>1
     for i=2:length(locs)
            ps(end+1)=(locs(i)-locs(i-1))*0.5*fp*6/300';
     end
end
