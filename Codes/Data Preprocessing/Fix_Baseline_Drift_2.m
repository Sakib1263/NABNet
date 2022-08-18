function [y] = Fix_Baseline_Drift_2(x)
     sig_max = max(x);        %Find the Signal Amplitude for Peak Detection Threshold
     sig_length = length(x);  %Find the Signal Length for Time Vector Creation
     sig_size = size(x);
     if sig_size(1) > 1
        x = x';
     end
     x = normalize(x,'range');
     [p,s,mu] = polyfit((1:numel(x)),x,20);
     f_y = polyval(p,(1:numel(x)),s,mu);
     y = x - f_y;
     % Fix the slight amplitude change due to baseline correction
     x_amp = max(x)-min(x);
     y_amp = max(y)-min(y);
     y = y*(x_amp/y_amp);
end