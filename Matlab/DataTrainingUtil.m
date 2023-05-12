classdef DataTrainingUtil
    methods(Static)
        % Assume x is a THz-TDS that incorporates a main peak, generate 7
        % frequency series ranged from 0.1 THz to 3 THz from x. The 7 sries
        % are transformed from the entire x, x truncated by 5 and 10 ps
        % from the front, x truncated by 5 and 10 ps from the end, and x
        % truncated by 5 and 10 ps from the front and the end. 
        %
        % f the frequencies of the series
        % xf a matrix that contains the 7 series of frequencies
        function [f, xf] = generateFrequencySeries(t, x)
            % The length of the frequency series ranged from 0.1 to 3
            % THz，where most of the absorption is at.
            lowLimit = 0.1;
            highLimit = 3;
            % The duration is 100 ps and the increments of frequency per
            % THz is 1 / (1/100) = 100
            duration = 100;
            seriesSize = (highLimit - lowLimit) * duration;
            xf = zeros(seriesSize, 7);

            % The entire x
            [f, xf(:, 1)] = FrequencyAnalysisUtil.convertToFrequency(t, x, duration, true, lowLimit, highLimit);
            
            % Truncate 5 ps from the beginning
            [tt, xt] = FrequencyAnalysisUtil.truncateTimeSeries(t, x, t(1) + 5, t(end));
            [~, xf(:, 2)] = FrequencyAnalysisUtil.convertToFrequency(tt, xt, duration, true, lowLimit, highLimit);
            
            % Truncate 10 ps from the beginning
            [tt, xt] = FrequencyAnalysisUtil.truncateTimeSeries(t, x, t(1) + 10, t(end));
            [~, xf(:, 3)] = FrequencyAnalysisUtil.convertToFrequency(tt, xt, duration, true, lowLimit, highLimit);
            
            % Truncate 5 ps from the end
            [tt, xt] = FrequencyAnalysisUtil.truncateTimeSeries(t, x, t(1), t(end) - 5);
            [~, xf(:, 4)] = FrequencyAnalysisUtil.convertToFrequency(tt, xt, duration, true, lowLimit, highLimit);
            
            % Truncate 10 ps from the end
            [tt, xt] = FrequencyAnalysisUtil.truncateTimeSeries(t, x, t(1), t(end) - 10);
            [~, xf(:, 5)] = FrequencyAnalysisUtil.convertToFrequency(tt, xt, duration, true, lowLimit, highLimit);
            
            % Truncate 5 ps from the beginning and the end
            [tt, xt] = FrequencyAnalysisUtil.truncateTimeSeries(t, x, t(1) - 5, t(end) - 5);
            [~, xf(:, 6)] = FrequencyAnalysisUtil.convertToFrequency(tt, xt, duration, true, lowLimit, highLimit);
            
            % Truncate 10 ps from the beginning and the end
            [tt, xt] = FrequencyAnalysisUtil.truncateTimeSeries(t, x, t(1) - 10, t(end) - 10);
            [~, xf(:, 7)] = FrequencyAnalysisUtil.convertToFrequency(tt, xt, duration, true, lowLimit, highLimit);
        end
    end
end