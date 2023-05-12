classdef ImageDataUtil
    methods(Static)
        % Unpack a file of serialized data and output the x, y, and THz
        % time series of each pixel of the image
        %
        % x1 x locations of the image
        % y1 y locations of the image
        % timeSeriesList [THz1,THz2,THz3, ...]组成的M行N列数组，列数由不同的坐标位置以及
        %对应的THz信号组成，用于三维切片成像所用的数据；
        % 
        % The directory of the data file
        % nda 单次总采样点数 
        function [x1, y1, timeSeriesList] = unpackData(fileName, nda)
            dat1= ImageDataUtil.readSerializedData(fileName);% 数据读入，Data Importing
            marker_vec=[163 163 165 165];% 标记位矢量,Marker Vector to Determine
            tn35=strfind(dat1,marker_vec);%求标记位位置，Position of marker vector
            %去除漏采的数据点，该方法虽然简洁但是无法打包，只能用笨办法
            dist=diff(tn35);% 求两个相邻标记位之间的点数 Find the distance between neighbor position of marker vectors
            pos_miss=find(dist~=2*nda+20);%如果点数不等于2nda+20,表明出现漏采 Find the missing points if the distance between two neighbor position is smaller than 2*nda
            if isempty(pos_miss)==0
                tn35(pos_miss)=[];%将漏采的数据去除
            end
            N_wave=length(tn35)-1;%有用的太赫兹信号数目
            dat2=zeros(nda,N_wave);%原始太赫兹信号初始化
            timeSeriesList=zeros(nda,N_wave);%去背景后太赫兹信号的初始化
            x1=zeros(1,N_wave);%x的坐标
            y1=zeros(1,N_wave);%y的坐标
            t=(0:1:nda-1)';
            for i=1:N_wave
                ttda2=dat1(tn35(i)+4:tn35(i)+2*nda+3);
                dat2(:,i)=ttda2(1:2:end)+ttda2(2:2:end)*256;%太赫兹信号的解码
                OrgTHz_i=dat2(:,i);%原始太赫兹信号
                [para,~,mu]=polyfit(t,OrgTHz_i,10);%求解背景，利用十次多项式拟合
                Val_pf=polyval(para,t,[],mu);%拟合的背景值
                timeSeriesList(:,i)=(OrgTHz_i-Val_pf);%去背景后的真实太赫兹信号%位置解码
                x1(i)=(double(int32(dat1(tn35(i)+nda*2+4)+bitshift(dat1(tn35(i)+nda*2+5),8)+bitshift(dat1(tn35(i)+nda*2+6),16)+bitshift(dat1(tn35(i)+nda*2+7),24))))/1000;%第一个电机接口作为x轴
                y1(i)=(double(int32(dat1(tn35(i)+nda*2+8)+bitshift(dat1(tn35(i)+nda*2+9),8)+bitshift(dat1(tn35(i)+nda*2+10),16)+bitshift(dat1(tn35(i)+nda*2+11),24))))/1000;%第二个电机接口作为y轴
            end
            
            %当x,y坐标超过15000，而且不在15000附近时处理
            for i=1:N_wave
                if abs(x1(i)-15000)>1000 && x1(i)>15000
                    x1(i)=x1(i)-16777;
                end
                if abs(y1(i)-15000)>1000 && y1(i)>15000
                    y1(i)=y1(i)-16777;
                end
            end
            %删除异常数据
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