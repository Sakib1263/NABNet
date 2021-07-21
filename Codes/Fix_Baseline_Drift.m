function [y] = Fix_Baseline_Drift(x)
    sig_max = max(x);        %Find the Signal Amplitude for Peak Detection Threshold
    sig_length = length(x);  %Find the Signal Length for Time Vector Creation
    sig_size = size(x);
    if sig_size(1) > 1
        x = x';
    end
    time = linspace(1,sig_length,sig_length)';
    
    % Find Average Peak to Peak Distance
    [~,locs] = findpeaks(x,'MinPeakProminence',sig_max/3);
    count = zeros(1, (length(locs)));
    for n = 1:(length(locs))
        locs_rounded = round(locs(1,n));
        count(1,n) = time(locs_rounded,1);
    end
    peak_dist = zeros(1, (length(count)-1));
    for n = 1:(length(count)-1)
        peak_dist(1,n) = count(n+1) - count(n);
    end
    peak_dist_median = median(peak_dist);        % Median or Mean; Median is better approximation since not affected by outliers
    
    % Window Size, Smaller is suitable for Higher Frequency Signals
    try
        baseline = movmin(x,peak_dist_median);                 % Get Lower Bounds
        p = polyfit(time',baseline,round(peak_dist_median/3)); % Polynomial Fit of the Baseline provides with better performance
        baseline_fit = polyval(p,time');                       % Polynomial Fit of the Baseline
        y_temp = x - baseline_fit;                             % Subtract the baseline from the original signal
        y = y_temp - min(y_temp);                              % Bring the Signal on the x-axis
    catch
        % Exception: Only if Smart Peak Detection Fails (e.g., EEG waves)
        [p,s,mu] = polyfit((1:numel(x)),x,20);
        f_y = polyval(p,(1:numel(x)),s,mu);
        y = x - f_y;
    end

     % Fix the slight amplitude change due to baseline correction
     x_amp = max(x)-min(x);
     y_amp = max(y)-min(y);
     y = y*(x_amp/y_amp);
end