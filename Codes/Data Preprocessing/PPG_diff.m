function [x1,x1d,x1dd,total_delay] = PPG_diff(x,sig_input_len,sig_output_len)
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
x1 = normalize(x,'range');
x1d = normalize((filter(d,x1)/dt),'range');
x1dd = normalize((filter(d,x1d)/dt),'range');
%
x1d = x1d(delay+1:end);
x1dd = x1dd(2*delay+1:end);
x1 = x1(1:end-2*delay);
%
x1 = x1(1:sig_output_len);
x1d = x1d(1:sig_output_len);
x1dd = x1dd(1:sig_output_len);
total_delay = 2*delay;
end