classdef FilterUtil
    methods(Static)
        % Apply EMD to x and remove the firt order
        % imf which is considered as high-frequency noise induced by the
        % viberation of the equipment.
        %
        % isSmooth if true, the profile of x is smooth and the extremas can
        % be fitted by cubic spline, otherwise, the profile is jagged and
        % piecewise-cubic Hermite interpolating polynomials is used to fit
        % the extremas.
        % TODO change the name to emdDenoise
        function y = hhtDenoise(x, isSmooth)
            interMode = 'spline';
            if ~isSmooth
                interMode = 'pchip';
            end
            [~, y] = emd(x,'MaxNumIMF',1, 'Interpolation',interMode);
        end
        
        % Remove noise from all the levels of the wavelet domain
        function y = waveletDenoise(x)
            % The level of wavelet denoising 
            n = nextpow2(length(x)) - 1;
            y = wdenoise(x,n,'Wavelet','coif4',...
                'DenoisingMethod','SURE',...
                'ThresholdRule','Hard',...
                'NoiseEstimate', 'LevelDependent');
        end
        
        % Remmove the noise using wavelet-packet decomposition
        function y = waveletPacketDenoise(x, level)
            % The level of wavelet denoising
            if nargin == 2
                n = level;
            else
                n = nextpow2(length(x)) - 1;
            end
            
            % Threshold of entropy
            thr = sqrt(2*log(n*log(n)/log(2)));
            y = wpdencmp(x,'s',n,'coif4','sure',thr,1);
        end
    end
end