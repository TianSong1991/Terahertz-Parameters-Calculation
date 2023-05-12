classdef ImageDataUtil
    methods(Static)
        % Unpack a file of serialized data and output the x, y, and THz
        % time series of each pixel of the image
        %
        % x1 x locations of the image
        % y1 y locations of the image
        % timeSeriesList [THz1,THz2,THz3, ...]��ɵ�M��N�����飬�����ɲ�ͬ������λ���Լ�
        %��Ӧ��THz�ź���ɣ�������ά��Ƭ�������õ����ݣ�
        % 
        % The directory of the data file
        % nda �����ܲ������� 
        function [x1, y1, timeSeriesList] = unpackData(fileName, nda)
            dat1= ImageDataUtil.readSerializedData(fileName);% ���ݶ��룬Data Importing
            marker_vec=[163 163 165 165];% ���λʸ��,Marker Vector to Determine
            tn35=strfind(dat1,marker_vec);%����λλ�ã�Position of marker vector
            %ȥ��©�ɵ����ݵ㣬�÷�����Ȼ��൫���޷������ֻ���ñ��취
            dist=diff(tn35);% ���������ڱ��λ֮��ĵ��� Find the distance between neighbor position of marker vectors
            pos_miss=find(dist~=2*nda+20);%�������������2nda+20,��������©�� Find the missing points if the distance between two neighbor position is smaller than 2*nda
            if isempty(pos_miss)==0
                tn35(pos_miss)=[];%��©�ɵ�����ȥ��
            end
            N_wave=length(tn35)-1;%���õ�̫�����ź���Ŀ
            dat2=zeros(nda,N_wave);%ԭʼ̫�����źų�ʼ��
            timeSeriesList=zeros(nda,N_wave);%ȥ������̫�����źŵĳ�ʼ��
            x1=zeros(1,N_wave);%x������
            y1=zeros(1,N_wave);%y������
            t=(0:1:nda-1)';
            for i=1:N_wave
                ttda2=dat1(tn35(i)+4:tn35(i)+2*nda+3);
                dat2(:,i)=ttda2(1:2:end)+ttda2(2:2:end)*256;%̫�����źŵĽ���
                OrgTHz_i=dat2(:,i);%ԭʼ̫�����ź�
                [para,~,mu]=polyfit(t,OrgTHz_i,10);%��ⱳ��������ʮ�ζ���ʽ���
                Val_pf=polyval(para,t,[],mu);%��ϵı���ֵ
                timeSeriesList(:,i)=(OrgTHz_i-Val_pf);%ȥ���������ʵ̫�����ź�%λ�ý���
                x1(i)=(double(int32(dat1(tn35(i)+nda*2+4)+bitshift(dat1(tn35(i)+nda*2+5),8)+bitshift(dat1(tn35(i)+nda*2+6),16)+bitshift(dat1(tn35(i)+nda*2+7),24))))/1000;%��һ������ӿ���Ϊx��
                y1(i)=(double(int32(dat1(tn35(i)+nda*2+8)+bitshift(dat1(tn35(i)+nda*2+9),8)+bitshift(dat1(tn35(i)+nda*2+10),16)+bitshift(dat1(tn35(i)+nda*2+11),24))))/1000;%�ڶ�������ӿ���Ϊy��
            end
            
            %��x,y���곬��15000�����Ҳ���15000����ʱ����
            for i=1:N_wave
                if abs(x1(i)-15000)>1000 && x1(i)>15000
                    x1(i)=x1(i)-16777;
                end
                if abs(y1(i)-15000)>1000 && y1(i)>15000
                    y1(i)=y1(i)-16777;
                end
            end
            %ɾ���쳣����
            for i=N_wave:1
                if x1(i)-x1(1)>1000
                    x1(i,:)=[];
                    y1(i,:)=[];
                    dat2(i,:)=[];
                end
            end
        end
        
        % Read the serialized data from the input directory
        function x = readSerializedData(fileName)
            if ~isfile(fileName)
                exception = MException('The file does not exist.',str);
                throw(exception)
            end

            x = readtable(fileName);
            x = x.(1);
            % strfind method only takes a row of values
            x = x';
        end 
        
        % Read data from a xls file, the first column is time,
        %
        % t the timings of all the series
        % x the values of all the series, i.e., N time series are in the
        % table, x = [y1, y2,..., yN]
        function [t, x] = readData(fileName)
            if ~isfile(fileName)
                exception = MException('The file does not exist.',str);
                throw(exception)
            end

            table = readtable(fileName);
            tableSize = size(table);
            t = table.(1);
            x = zeros(tableSize(1), tableSize(2) - 1);
            for i = 2 : tableSize(2)
                % The series in x could be discrepant
                x(:, i - 1) = table.(i);
            end
            
            % Remove the nan values in t and x
            xNanIndices = isnan(x);
            tNanIndices = isnan(t);
            removedIndices = any(xNanIndices');
            removedIndices = removedIndices | tNanIndices';
            t(removedIndices) = [];
            x(removedIndices, :) = []; 
        end     
    end
end