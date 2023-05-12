classdef InputArgumentsUtil
    methods(Static)
        % b if true, all the input arguments are of the same length
        function b = isSameLength(varargin)
            lengths = zeros(nargin, 1);
            for ii = 1 : nargin
                lengths(ii) = length(varargin{ii});
            end
            
            b = all(lengths == lengths(1));
        end
    end
end