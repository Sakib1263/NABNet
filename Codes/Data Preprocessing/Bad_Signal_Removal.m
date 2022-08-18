i = i+250;

A = normalize(ABP(:,i),'range');
B = normalize(PPG(:,i),'range');

sig_size_A = size(A);
sig_size_B = size(B);
if sig_size_A(1) > 1
    A = A';
end
if sig_size_B(1) > 1
    B = B';
end

time = linspace(1,1024,1024)';

[pks_ABP,locs_ABP,w1,p1] = findpeaks(A,'MinPeakDistance',0.1,'MinPeakProminence',0.2);
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
num_peaks_ABP = length(pks_ABP);
pks_ABP_1 = pks_ABP(2:end-1);
std_peaks_ABP = std(pks_ABP_1,1);

[pks_PPG,locs_PPG,w2,p2] = findpeaks(B,'MinPeakDistance',0.1,'MinPeakProminence',0.2);
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
num_peaks_PPG = length(pks_PPG);
pks_PPG_1 = pks_PPG(2:end-1);
std_peaks_PPG = std(pks_PPG_1,1);

subplot(2,1,1);
plot(time,A,locs_ABP,pks_ABP,'o');
title('ABP');
xlim([0 1024]);
grid on;
subplot(2,1,2);
plot(time,B,locs_PPG,pks_PPG,'o');
title('PPG');
xlim([0 1024]);
grid on;

if ((std_peaks_ABP > 0.12) || (std_peak_dist_ABP > 10) || (std_peaks_PPG > 0.12) || (std_peak_dist_PPG > 10) || (num_peaks_PPG < 6) || (num_peaks_ABP < 6))
    fprintf('Bad Signal\n');
    sgtitle('Decision: Bad Signal (Any one)');
else
    fprintf('Good Signal\n');
    sgtitle('Decision: Good Signal (Both)');
end

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
%%
Decision = Remove_Bad_Signals(B,A);
disp(Decision);