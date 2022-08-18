function [x1,x1d,x1dd,y1,y1_RNorm,z1,total_delay] = PPG_diff_PS_Correction(x,y,z,sig_input_len,sig_output_len)
% Construct Digital Filter
Nf = 50; 
Fpass = 20; 
Fstop = 100;
Fs = 1000;
d = designfilt('differentiatorfir','FilterOrder',Nf, ...
    'PassbandFrequency',Fpass,'StopbandFrequency',Fstop, ...
    'SampleRate',Fs);
delay = mean(grpdelay(d));
%
sample_num = linspace(1,sig_input_len,sig_input_len);
dt = sample_num(2)-sample_num(1);
%
x1 = normalize(normalize(x,'zscore'),'range');
y1 = normalize(normalize(y,'zscore'),'range');
z1 = normalize(normalize(z,'zscore'),'range');
%
sig_max_A = max(x1);  %Find the Signal Amplitude for Peak Detection Threshold
sig_max_B = max(y1);  %Find the Signal Amplitude for Peak Detection Threshold
%
sig_size_A = size(x1);
sig_size_B = size(y1);
sig_size_C = size(z1);
%
if sig_size_A(1) > 1
    x1 = x1';
end
if sig_size_B(1) > 1
    y1 = y1';
end
if sig_size_C(1) > 1
    z1 = z1';
end
% Find Average Peak to Peak Distance
[~,locs1] = findpeaks(x1,'MinPeakProminence',sig_max_A/15);
[~,locs2] = findpeaks(y1,'MinPeakProminence',sig_max_B/15);
num_locs1 = length(locs1);
num_locs2 = length(locs2);
%
if num_locs1 < num_locs2
    num_locs = num_locs1;
else
    num_locs = num_locs2;
end
%
locs_diff_arr = zeros(1, num_locs);
for n = 1:num_locs
    locs_diff = locs1(n)-locs2(n);
    locs_diff_arr(1,n) = locs_diff;
end
%
if locs_diff_arr(2) == 0
    locs_diff_arr(2) = 1;
end
if abs(locs_diff_arr(2)) >= (sig_input_len - 1100)
    locs_diff_arr(2) = -((sig_input_len - 1100)-1);
end
% Align PPG, ABP and ECG Peaks
if locs1(2) > locs2(2)
    x1 = x1(abs(locs_diff_arr(2)):length(x1));
    y = y(1:length(y)-abs(locs_diff_arr(2)));
    z1 = z1(abs(locs_diff_arr(2)):length(z1));  % Align ECG
elseif locs1(2) < locs2(2)
    x1 = x1(1:length(x1)-abs(locs_diff_arr(2)));
    y = y(abs(locs_diff_arr(2)):length(y));
end
% Find PPG Derivatives and Filter Distortions
x1d = normalize(normalize((filter(d,x1)/dt),'zscore'),'range');
x1dd = normalize(normalize((filter(d,x1d)/dt),'zscore'),'range');
% Align APG and VPG to PPG
x1d = x1d(delay+1:end);
x1dd = x1dd(2*delay+1:end);
x1 = x1(1:end-2*delay);
% Cut all signals to the desired signal length
x1 = x1(1:sig_output_len);
x1d = x1d(1:sig_output_len);
x1dd = x1dd(1:sig_output_len);
y1 = y(1:sig_output_len);
y1_RNorm = normalize(normalize(y1,'zscore'),'range');
z1 = z1(1:sig_output_len);
total_delay = 2*delay;
end