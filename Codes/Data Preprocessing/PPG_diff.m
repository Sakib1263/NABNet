function [x1,x1d,x1dd,y1,y1_RNorm,z1,total_delay] = PPG_diff(x,y,z,sig_input_len,sig_output_len)
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
x1 = normalize(normalize(x,'zscore'),'range');
z1 = normalize(normalize(z,'zscore'),'range');
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