% This class is to calculate the refractive index and the absorption rate
% of a sample measured by the THz system. An initial measurement without the  
% sample has to be applied before the ones with the sample. 
classdef SampleProperties
    properties
        % The discrete times of the reference time-domain spectrum
        refT
        % The amplitudes of the reference time-domain spectrum
        refX
        % The discrete times of the time-domain spectrum of the sample
        sampT
        % The amplitudes of the time-domain spectrum of the sample
        sampX
        % The depth of the sample, unit mm, usually between 1 to 2 mm
        d{mustBeNumeric}
        % If true compensate the phase transition function by the
        % difference between the estimated phase transition and that with
        % no dispersion
        compPhase
        % If true, use the deconvolutional denooising to eliminate the
        % reflection peaks
        denoise
        % The ratio of the frequency amplitude of the sample and the
        % frequency amplitude of the empty mining
        ampTransFunction
        % The difference between the phase of the frequency spectrum of the
        % sample and that of the empty mining
        phaseTransFunction
        % The discrete frequencies of the smaple's frequency-domain
        % spectrum, unit THz
        f
    end
    methods
        function obj = SampleProperties(refT, refX, sampT, sampX, d, compPhase, denoise)
            if nargin == 5
                obj.refT = refT;
                obj.refX = refX;
                obj.sampT = sampT;
                obj.sampX = sampX;
                obj.d = d;
                obj.compPhase = false;
                obj.denoise = false;
            elseif nargin == 6
                obj.refT = refT;
                obj.refX = refX;
                obj.sampT = sampT;
                obj.sampX = sampX;
                obj.d = d;
                obj.compPhase = compPhase;
                obj.denoise = false;
            elseif nargin == 7
                obj.refT = refT;
                obj.refX = refX;
                obj.sampT = sampT;
                obj.sampX = sampX;
                obj.d = d;
                obj.compPhase = compPhase;
                obj.denoise = denoise;
            else
                error('The number of input arguments shoulb be five.')
            end
            
            % Calculate the transmission function
            [obj.ampTransFunction, obj.phaseTransFunction, obj.f] = obj.calTransFunction();
        end
        
        % Calculate the sample's refractive index 
        function sampRefraction = calSampleRefraction(obj)
            % The light spped in vacuum, unit m/s
            c = 3e8;            
            sampRefraction = obj.phaseTransFunction * c;
            sampRefraction = sampRefraction / (2 * pi) ./ obj.f / obj.d / 1e9; 
            sampRefraction = sampRefraction + 1;
        end
        
         % Calculate the smaple's absorption rate
        function sampAbsortption = calSampleAbsorption(obj)
            % Calculate the corrected refraction rate
            n = obj.getCorrectedRefraction();
            
            % Absorptioin rate, cited from 'Lu 2013'
            sampAbsortption = log(4 * n ./ obj.ampTransFunction ./ (1 + n).^2);
            
            % Unit cm^-1
            sampAbsortption = 2 / (obj.d * 0.1) * sampAbsortption; 
        end
    end
    
    methods(Access = private)
        % Get the phase and the amplitude of a frequency spectrum by
        % converting from the time spectrum
        function [phase, amp] = getPhaseAndAmplitude(obj, x, f)
            if obj.denoise
                xf = FrequencyAnalysisUtil.removeReflectionPeaks(obj.refT, f, x);
            else
                xf = fft(x);
            end
            phase = angle(xf);
            amp = abs(xf);
        end
        
        % Make linear fitting to the sample's refraction rate
        function n = getCorrectedRefraction(obj)
            % Fit the phase transition from 0.3 THz to 0.5 THz
            lowLimit = 0.3;
            upLimit = 0.5;
            range = obj.f > lowLimit & obj.f < upLimit;
            slope = obj.phaseTransFunction(range) \ obj.f(range);
            
            % Light speed m/s
            c = 3.0e8;
            
            % c / (d * 2pi)
            factor = c / obj.d / (2 * pi) / 1e9;
            n = factor * slope + 1;
        end 
        
        % Calculate the sample's transmission function using the time-domain
        % spectrum of the system without the sample and that with the
        % sample 
        function [ampTrans, phaseTrans, f] = calTransFunction(obj)
            % The default duration is 100 ps
            duration = 100;
            sampDuration = obj.sampT(end) - obj.sampT(1);
            refDuration = obj.refT(end) - obj.refT(1);

            % Concatenate the time series to the longest duration
            duration = max([duration, sampDuration, refDuration]);
            [newSampT, newSampX] = FrequencyAnalysisUtil.concatenateTimeSeries(obj.sampT, obj.sampX, duration);
            [~, newRefX] = FrequencyAnalysisUtil.concatenateTimeSeries(obj.refT, obj.refX, duration);
            f = FrequencyAnalysisUtil.calRangeOfFrequency(newSampT, length(newSampT), false);
            
%             [sampPhase, sampAmp] = obj.getPhaseAndAmplitude(newSampX, f); 
%             [refPhase, refAmp] = obj.getPhaseAndAmplitude(newRefX, f);
            refXf = fft(newRefX);
            sampXf = fft(newSampX);
            xf = sampXf ./ refXf;
            ampTrans = abs(xf);
            phaseDiff = angle(xf);
%             ampTrans = sampAmp ./ refAmp;
            
            
            
            % The lower limit of the effective signal
            sigLowLimit = 0.1;
            slIndices = f >= sigLowLimit;
%             phaseDiff = sampPhase - refPhase;
            phaseDiff = phaseDiff(slIndices);
            ampTrans = ampTrans(slIndices);
            f = f(slIndices);
            
            % Make the sawtooth-shape angle series continuous
            phaseTrans = -unwrap(phaseDiff);
            
            if obj.compPhase
                % Calculate the offset between the estimated phase and that
                % without dispersion 
                delay = obj.calTimeDelay();
                noDispPhaseTrans = 2 * pi * f * delay;

                % The portion of the highest SNR
                highSnrIndices = f > 0.3 & f < 0.5;
                offset = round(mean(phaseTrans(highSnrIndices)...
                    - noDispPhaseTrans(highSnrIndices)) / pi) * pi;
                phaseTrans = phaseTrans - offset;
            end
        end
        
        % Calculate the time delay induced by the sample
        function delay = calTimeDelay(obj)
            [~, refPeakIndex] = max(obj.refX);
            refPeakTime = obj.refT(refPeakIndex);
            [~, sampPeakIndex] = max(obj.sampX);
            sampPeakTime = obj.sampT(sampPeakIndex);
            delay = sampPeakTime - refPeakTime;
        end
    end
end