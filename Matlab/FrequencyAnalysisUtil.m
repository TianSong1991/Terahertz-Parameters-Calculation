classdef FrequencyAnalysisUtil
    methods(Static)
        % Transform the time domain series to frequency domain, applicable
        % if x contains multiple series
        %
        % f the equally distanced steps within the range of frequency
        % xf the amplitudes of freqnencies
        % duration concaternate the time series to the duration to increase
        % the precision 
        % isInDB if convert the amplitudes of frequency to DB 
        % fs the start of the range of frequency
        % ft the end of the range of frequency
        function [f, xf] = convertToFrequency(t, x, duration, isInDB, fs, ft, denoise)
            switch nargin
                case 2
                    % The default duration is 100ps, and thus the increment
                    % of the frequency is 0.01THz
                    [f, xf] = FrequencyAnalysisUtil.timeToFrequency(t, x, 100, false);
                case 3
                    [f, xf] = FrequencyAnalysisUtil.timeToFrequency(t, x, duration, false);
                case 4
                    [f, xf] = FrequencyAnalysisUtil.timeToFrequency(t, x, duration, false);
                    if isInDB
                        xf = FrequencyAnalysisUtil.amplitudeToDb(xf);
                    end
                case 6
                    [f, xf] = FrequencyAnalysisUtil.timeToFrequency(t, x, duration, false);
                    if isInDB
                        xf = FrequencyAnalysisUtil.amplitudeToDb(xf);
                    end
                    [f, xf] = FrequencyAnalysisUtil.selectRangeOfFrequency(f, xf, fs, ft);
                case 7
                    [f, xf] = FrequencyAnalysisUtil.timeToFrequency(t, x, duration, denoise);
                    if isInDB
                        xf = FrequencyAnalysisUtil.amplitudeToDb(xf);
                    end
                    [f, xf] = FrequencyAnalysisUtil.selectRangeOfFrequency(f, xf, fs, ft);
            end
        end
            
        % see convertToFrequency
        function [f, xf] = timeToFrequency(t, x, duration, denoise)
            [~, xConcat] = FrequencyAnalysisUtil.concatenateTimeSeries(t, x, duration);
            timeLength = size(xConcat, 1);
            
            % The unit of frequency is ps
            f = FrequencyAnalysisUtil.calRangeOfFrequency(t, timeLength, true);

            xf = zeros(size(xConcat));
            
            % If the x contains multiply channels, iterate through all the
            % channnels
            for ii = 1 : size(xConcat, 2)
                if denoise
                    xfSeries = FrequencyAnalysisUtil.removeReflectionPeaks(t, f, xConcat);
                else
                    xfSeries = fft(xConcat(:, ii));
                end
                
                %除以N再乘以2才是真实幅值，N越大，幅值精度越高
                xfSeries = xfSeries / timeLength * 2;            
                xfSeries = abs(xfSeries);
                
                % Normalization
                xfSeries = xfSeries / max(xfSeries);
                xf(:, ii) = xfSeries;
            end
        end
        
        % Concatenate time series to a specified duration.
        function [tc, xc] = concatenateTimeSeries(t, x, duration)
            % The original duration of time 
            origDuration = t(end) - t(1);
            % The average of all the increments of timming
            deltaT = FrequencyAnalysisUtil.calAverageTimeStep(t);
            timeLength = length(t);
            
            % Concatenate time series if it is shorter than duration
            if origDuration < duration
                concatLength = round((duration - origDuration) / deltaT);
                newTimeLength = timeLength + concatLength;
                tc = t(1) + (0 : newTimeLength - 1)' * deltaT;
                xc = zeros(newTimeLength, size(x, 2));
                xc(1 : timeLength, :) = x;
            else
                tc = t;
                xc = x;
            end
        end
        
        % Calculate the average time step of a series
        function dt = calAverageTimeStep(t)
            dt = mean(t(2 : end) - t(1 : end - 1));
        end
        
        % Truncate a time series from the start, ts, to the end, te.
        function [tc, xc] = truncateTimeSeries(t, x, ts, te)            
            indices = t >= ts & t <= te;
            tc = t(indices);
            xc = x(indices);
        end
        
        % Calculate the range of frequency from the time step of t. 
        %
        % n the number steps of the sampling
        %
        % f the discrete values of frequency
        % df the step size of f
        % isFromZero if the frequency starts from 0
        function [f, df] = calRangeOfFrequency(t, n, isFromZero)
            % The average of all the increments of timming.
            dt = FrequencyAnalysisUtil.calAverageTimeStep(t);
            fRange = 1 / dt;
            df = fRange / n;
            
            if isFromZero
                f = (0 : n - 1)' * df;
            else 
                f = (1 : n)' * df;
            end
        end
            
        % Convert the amplitude from the absolute value to the unit of dB
        function xfDB = amplitudeToDb(xf)
            xfDB = 20 * log10(xf);
        end
        
        % Select the range of frequency from f, starting from fs and
        % endding at ft. The method is applicable if xf contains multiple
        % series
        function [fp, xfp] = selectRangeOfFrequency(f, xf, fs, ft)
            range = f >= fs & f <= ft;
            fp = f(range);
            xfp = xf(range, :);
        end
        
        % Return the last Imf of x by empirical mode decomposition (EMD)
        %
        % isSmooth if true, the profile of x is smooth and the extremas can
        % be fitted by cubic spline, otherwise, the profile is jagged and
        % piecewise-cubic Hermite interpolating polynomials is used to fit
        % the extremas.
        function y = calLastImf(x, isSmooth)
            interMode = 'spline';
            if ~isSmooth
                interMode = 'pchip';
            end
            imf = emd(x, 'Interpolation',interMode);
            y = imf(:, size(imf, 2));
        end
        
        % See calLastImf
        function y = calFirstImf(x, isSmooth)
            interMode = 'spline';
            if ~isSmooth
                interMode = 'pchip';
            end
            y = emd(x, 'MaxNumIMF',1, 'Interpolation',interMode);
        end
        
        % Calculate the time delay of the peak induced by the sample
        % The length of the reference series can be differnt from the
        % smaple series.
        function pDelay = calTimeDelay(refT, refX, sampT, sampX)
            [~, refPeakIndex]=max(refX);
            [~, sampPeakIndex]=max(sampX);
            
            %给出参考信号和样品信号时域波形间的时间延迟，由两波形峰值位置测量得, unit ps
            pDelay = sampT(sampPeakIndex) - refT(refPeakIndex);
        end 
        
        % Calculate the noise of signal ratio of a measurement. It is
        % required by the modified Wiener filter
        function nsr = calNsr(refX, noiseX)
            nsr = 1 / snr(refX, noiseX);
        end
        
        % Apply the deconvolution denoising to the inputt signal.
        % Referenced from 'Naftaly 2006'
        %
        % y the denoised frequency domain amplitudes converted from the
        % original signal
        function y = removeReflectionPeaks(t, f, x)
            ihhtf = ImprovedHhtFilter(t, x);
            [~, ~, ~, mPeakIndex] = ihhtf.findMainPeak();
            reflectPeaks = ihhtf.findReflectionPeaks();
            
            % The demoninator to justify the effect of reflection peaks
            deconvFactor = 1;
            
            for ii = 1 : size(reflectPeaks, 1)
                refPeakIndex = reflectPeaks(ii, 3);
                tDiff = t(refPeakIndex) - t(mPeakIndex);
                refMainRatio = x(refPeakIndex) / x(mPeakIndex);
                deconvFactor = deconvFactor + refMainRatio * exp(-2i * pi * f * tDiff);
            end
            
            xf = fft(x);
            y = xf ./ deconvFactor;
        end
    end
end