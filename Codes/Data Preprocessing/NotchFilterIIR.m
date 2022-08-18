function [y] = NotchFilterIIR(x,cutoff_frequency,signal_sampling_frequency,q)
wo = cutoff_frequency/(signal_sampling_frequency/2);  
bw = wo/q;
[b,a] = iirnotch(wo,bw);
y = filter(b,a,x);
end
