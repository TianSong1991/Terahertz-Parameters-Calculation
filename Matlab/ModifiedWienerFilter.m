% The class is designed to apply a Wiener-like filter to THz time series. 
% The filter is designed by convolving a Wiener filter to a time window
% which follows exp(-t^2/ sigma^2), sigma is the width of the reflected THz
% wave from the sample's surface
classdef ModifiedWienerFilter
    properties
        % The FWHM of the reflected THz wave
        fwhm {mustBeNumeric}
        % The noise to signal ratio of the reflected THz wave
        nsr {mustBeNumeric}
        % The timings of the THz wave, unit ps
        recT
        % The amplitudes of the reference signal
        recX
        % The amplitudes of the reflected signal from sample's surface 
        recY
        
    end
    methods
        function obj = ModifiedWienerFilter(recT, recX, recY, fwhm, nsr)
            if nargin == 5
                obj.recT = recT;
                obj.recX = recX;
                obj.recY = recY;
                obj.fwhm = fwhm;
                obj.nsr = nsr;
            else
                error('The number of inputs should be 5');
            end
        end
        
        % Use the Gaussian-shape window to remove the distortion of
        % reflected, the unit of the frequency THz
        % peak of the time series
        function mwf = applyModifiedWienerFilter(obj)
            win = obj.getTimeWindow();
            wf = obj.getWienerFilter();
            xFreq = fft(obj.recX);
            yFreq = fft(obj.recY);
            % Cite from 'Nondestructive testing of rubber materials based on
            %terahertz time-domain spectroscopy technology'
            mwfFreq = yFreq .* wf .* win .* xFreq;
            mwf = ifft(mwfFreq, 'symmetric');
        end
        
        % The window in the time domain is a half Gaussian shape function 
        % that separate the signal portion of the time series from the rest 
        % of it. 
        %
        % win The window in the frequency domain
        function win = getTimeWindow(obj)
            freq = obj.getFrequency();
            sig = obj.fwhm / 2 / sqrt(2 * log(2));
            % The Fourier transform of the original Gaussian-shape window
            win = exp(-2 * pi^2 * sig^2 * freq);
        end
        
        % Estimate the frquency domain Wiener filter from the referenced
        % signal and SNR
        %
        % wf The profile of the Wiener filter 
        function wf = getWienerFilter(obj)
            xFrec = fft(obj.recX);
            wf = conj(xFrec) ./ (abs(xFrec).^2 + obj.nsr);
        end
        
        % Use the step size of the time series to calculate the coordinates
        % of the frequency domain
        function freq = getFrequency(obj)
            % The average step size of the time series, unit ps
            dt = mean(diff(obj.recT));
            n = length(obj.recT);
            % The coordinates of the frequency, unit THz
            freq = 1 / dt * (0 : n - 1) / n;
            freq = freq';
        end
    end
end