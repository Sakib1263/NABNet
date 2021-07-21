function [Decision] = Remove_Bad_Signals(PPG,ABP,sig_length)
    A = normalize(PPG,'range');
    B = normalize(ABP,'range');
    
    % Transpose the Signals if they are in column format, necessary for peak detection
    sig_size_A = size(A);
    sig_size_B = size(B);
    if sig_size_A(1) > 1
        A = A';
    end
    if sig_size_B(1) > 1
        B = B';
    end

    time = linspace(1,sig_length,sig_length)';
    
    [pks_PPG,locs_PPG] = findpeaks(A,'MinPeakDistance',0.1,'MinPeakProminence',0.2);
    count = zeros(1, (length(locs_PPG)));
    for n = 1:(length(locs_PPG))
        locs_rounded2 = round(locs_PPG(1,n));
        count(1,n) = time(locs_rounded2,1);
    end
    peak_dist_PPG = zeros(1, (length(count)-1));
    for n = 1:(length(count)-1)
        peak_dist_PPG(1,n) = count(n+1) - count(n);
    end
    std_peak_dist_PPG = std(peak_dist_PPG,1);
    std_peaks_PPG = std(pks_PPG,1);
    num_peaks_PPG = length(pks_PPG);

    [pks_ABP,locs_ABP] = findpeaks(B,'MinPeakDistance',0.1,'MinPeakProminence',0.2);
    count = zeros(1, (length(locs_ABP)));
    for n = 1:(length(locs_ABP))
        locs_rounded1 = round(locs_ABP(1,n));
        count(1,n) = time(locs_rounded1,1);
    end
    peak_dist_ABP = zeros(1, (length(count)-1));
    for n = 1:(length(count)-1)
        peak_dist_ABP(1,n) = count(n+1) - count(n);
    end
    std_peak_dist_ABP= std(peak_dist_ABP,1);
    std_peaks_ABP = std(pks_ABP,1);
    num_peaks_ABP = length(pks_ABP);

    if ((std_peaks_ABP > 0.12) || (std_peak_dist_ABP > 12) || (std_peaks_PPG > 0.12) || (std_peak_dist_PPG > 12) || (num_peaks_PPG < 6) || (num_peaks_ABP < 6))
        Decision = 0;
    else
        Decision = 1;
    end
end