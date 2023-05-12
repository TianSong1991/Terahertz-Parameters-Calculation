% This modified HHT filter is applicable to transmissive systems. For
% reflection systems, the second reflection peak may be stronger than the
% first one and thus disrupt the logic of this class.
classdef ImprovedHhtFilter < HhtFilter
    methods
        % See findReflectionPeaks@Super(obj)
        function newRps = findReflectionPeaks(obj)
            % Calculate the sampling rate of the time series
            dt = FrequencyAnalysisUtil.calAverageTimeStep(obj.t);
            fs = 1 / dt;
            
            % Extract the first imf from the denoised signal, because the first imf
            % contains the hig-frequency component of the signal which complies with
            % the peaks
            imf = emd(obj.xt, 'MaxNumIMF',1);

            % Calculate the Hilbert spectrum of the first imf
            hs = hht(imf, fs);

            % Find the non-zero elements in hs
            [fIndices, tIndices, values] = find(hs);

            % Remove the elements which values are too low
            % TODO define the ratio of 0.001 in the input
            [fIndices, tIndices] = obj.removeSmallValues(fIndices, tIndices, values, 0.001);
            
            % Find the reflected peaks. For the transmissive sampling, the reflectd
            % peaks follow the main peak
            rps = findReflectionPeaks@HhtFilter(obj);
            [~, startIndex, endIndex] = obj.findMainPeak();
            
            % Find the frequency components in the main peak
            mpFreqIndices = obj.findIndicesOfFrequencies(fIndices, tIndices, startIndex, endIndex);
            
            % Find the maximum frequency component within the main peak 
            mpMaxFreqIndex = max(mpFreqIndices);

            % The threshold to decide that a high frequency component exists in the
            % reflected peak, i.e., the maximum frequency the main peak contains is 10
            % and the maximum frequency a reflected peak contains is 5, then the
            % reflected peak is considered to be a effective peak. 
            % TODO define the ratio 0.5 in the input
            rpFreqThreshold = 0.5 * mpMaxFreqIndex;
            
            % Find the effective reflection peaks
            newRps = obj.findEffectivePeaks(rps, fIndices, tIndices, endIndex, rpFreqThreshold);
        end
    end
    
    methods(Access = private)
        % Remove the elements which values are too low
        %
        % ratio the ratio of the threshold of the values to be removed and
        % the maximum value of the Hilbert spectrum
        function [newFIndices, newTIndices] = removeSmallValues(obj, fIndices, tIndices, values, ratio)
            minValue = ratio * max(values);
            vIndices = values > minValue;
            newFIndices = fIndices(vIndices);
            newTIndices = tIndices(vIndices);
        end
        
        % Find the effective reflection peaks by comparing the maximum
        % frquency contained by each estimated peak and the threshold of
        % frequency. 
        %
        % peaks the indices of the reflection peaks determined by the correlation 
        % endInex the end position of main peak
        function newPeaks = findEffectivePeaks(obj, peaks, fIndices, tIndices, endIndex, threshold)
            newPeaks = zeros(size(peaks));
            numNewPeaks = 0;

            for ii = 1 : size(peaks, 1)
                newFIndices = fIndices(tIndices > peaks(ii, 1) & tIndices < peaks(ii, 2));
                if ~isempty(newFIndices)
                    maxFIndex = max(newFIndices);
                    
                    % If the reflected peak follows the main peak and contains a maximum
                    % frequency that is greater than the threshold, then it is a effecive
                    % peak. 
                    if peaks(ii, 1) > endIndex && maxFIndex > threshold
                        numNewPeaks = numNewPeaks + 1;
                        newPeaks(numNewPeaks, :) = peaks(ii, :);
                    end
                end
            end
            
            % Remove zeros from newPeaks
            if numNewPeaks < size(peaks, 1)
                newPeaks(numNewPeaks + 1 : end, :) = [];
            end
        end
        
        % Given the start and end positions of peaks, find the indices within the peak
        function freqIndices = findIndicesOfFrequencies(obj, fIndices, tIndices, startIndex, endIndex)
             freqIndices = fIndices(tIndices > startIndex & tIndices < endIndex);
        end
    end
end
