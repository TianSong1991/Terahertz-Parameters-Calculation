classdef HhtFilter
    properties
        % The width of the main peak of the THz signal
        mainPeakWidth{mustBeNumeric}
        % The timings of the THz time-domain signal
        t
        % The THz time-domain signal to process
        xt
        % The threshold of correlation, above which a maxima is determined as
        % a reflection of the main peak
        corrThreshold{mustBeNumeric}
        % If the profile of the input time series is smooth, use spline to
        % fit the maximas,otherwise use piecewise-cubic Hermite.
        isSmooth
    end
    methods
        function obj = HhtFilter(t, x, mPeakWidth, cThreshold, s)
            switch nargin 
                case 2
                    obj.t = t;
                    obj.xt = x;
                    % The defualt value is cited from '太赫兹光谱数据处理及定量分析研究'
                    obj.mainPeakWidth = 20;
                    % The default value is taken from the sample data
                    obj.corrThreshold = 0.1;
                    obj.isSmooth = true;
                case 3
                    obj.t = t;
                    obj.xt = x;
                    obj.mainPeakWidth = mPeakWidth;
                    obj.corrThreshold = 0.1;
                    obj.isSmooth = true;
                case 4
                    obj.t = t;
                    obj.xt = x;
                    obj.mainPeakWidth = mPeakWidth;
                    obj.corrThreshold = cThreshold;
                    obj.isSmooth = true;
                case 5
                    obj.t = t;
                    obj.xt = x;
                    obj.mainPeakWidth = mPeakWidth;
                    obj.corrThreshold = cThreshold;
                    obj.isSmooth = s;
                otherwise
                    error('The number of input arguments should be less than 5.');
            end    
        end
        
        % Find the range of the main peak, which is from mainPeakWidth prior
        % to the peak to mainPeakWidth after the valley
        %
        % y the discrete values of the main peak
        % startIndex the start position of the main peak
        % endIndex the end position of the main peak
        % peakIndex the position of the peak
        % TODO the first reflection peak can be lower than the second one
        % in a reflection system
        function [y, startIndex, endIndex, peakIndex] = findMainPeak(obj)
            [~, peakIndex] = max(obj.xt);
            [~, valleyIndex] = min(obj.xt);
            
            % The peak is before the valley
            startIndex = peakIndex - obj.mainPeakWidth;
            endIndex = valleyIndex + obj.mainPeakWidth;
            
            % The valley is before the peak
            if peakIndex > valleyIndex
                startIndex = valleyIndex - obj.mainPeakWidth;
                endIndex = peakIndex + obj.mainPeakWidth;
            end
            
            if startIndex  < 1
                startIndex = 1;
            end
            
            if endIndex > length(obj.xt)
                endIndex = length(obj.xt);
            end
            
            y = obj.xt(startIndex : endIndex);
        end
        
        % Find the starts and the ends of all the reflection peaks. The
        % main peak is correlated with the rest of the time series to
        % locate the reflected peaks. In the reflective sampling, the
        % highest peak could be the the first reflected peak, which is
        % second in the sequence. 
        %
        % y see writeReflectionPeaks
        function y = findReflectionPeaks(obj)
            [xMp, mStartIndex, mEndIndex] = obj.findMainPeak();
            startToEnd = mEndIndex - mStartIndex;

            % Substitue the portion of the main peak to zeros
            xRest = obj.xt;
            xRest(mStartIndex : mEndIndex) = 0;
            
            % Concatenate zeros to the end of the main peak to make it the same 
            % length as the xRest. It is required by the xcorr method.
            xMpComp = zeros(length(xRest), 1);
            xMpComp(1 : length(xMp)) = xMp;
                      
            v= ver('MATLAB'); 
            if v.Release == "(R2020a)" || v.Release == "(R2020b)" 
                [xCorr, lags] = xcorr(xRest, xMpComp, 'normalized');
            else
                % For versions before 2020a
                [xCorr, lags] = xcorr(xRest, xMpComp, 'coeff');
            end
            
            xCorr = xCorr(lags >= 0);
            
            % The minimum distance between two consecutive peaks
            minPeakDistance = obj.mainPeakWidth * 4;
            [~, corrPeakIndices] = findpeaks(xCorr, 'MinPeakHeight', obj.corrThreshold, 'MinPeakDistance',minPeakDistance);
           
            % Write the starts and the ends of the reflection peaks into y
            y = obj.writeReflectionPeaks(corrPeakIndices, startToEnd);
        end
        
        % Use HHT filter to smooth the portion of reflected peaks, cited
        % from 'Lu 2013'
        function xf = apply(obj)
            refPeakIndeices = obj.findReflectionPeaks();
            xf = obj.xt;
            
            % No reflected peak is found
            if isempty(refPeakIndeices) 
                xf = obj.xt;
            else
                for i = 1 : size(refPeakIndeices, 1)
                    pStart = refPeakIndeices(i, 1);
                    pEnd = refPeakIndeices(i, 2);
                    peak = obj.xt(pStart : pEnd);
                    
                    % Take the last Imf as the recovered signal
                    smoothedPeak = FrequencyAnalysisUtil.calLastImf(peak, true);
                    xf(pStart : pEnd) = smoothedPeak;
                end
            end                 
        end
    end
    
    methods(Access = private)
        % Write the starts and ends of the reflection peaks into y, in
        % which y(:, 1) are the starts, y(:, 2) are the ends, and y(:, 3)
        % are the peaks of the refleciton peaks
        %
        % corrPeakIndices the start points of the reflection peaks
        % startToEnd the distance between the start and the end of the main
        % peak, and the span of the reflection peaks are assumed to be the
        % same.
        function y = writeReflectionPeaks(obj, corrPeakIndices, startToEnd)
            nRefPeaks = length(corrPeakIndices);
            
            if nRefPeaks == 0
                y = [];
            else
                y = zeros(nRefPeaks, 3);
                for i = 1 : length(corrPeakIndices)
                      % The location of a correlation peak represent the
                      % beginning of a reflected peak
                      y(i, 1) = corrPeakIndices(i);
                      y(i, 2) = corrPeakIndices(i) + startToEnd;
                      if y(i, 2) > length(obj.xt)
                          y(i, 2) = length(obj.xt);
                      end
                      
                      [~, mIndex] = max(obj.xt(y(i, 1) : y(i, 2)));
                      y(i, 3) = mIndex;
                end
            end
        end
    end
end