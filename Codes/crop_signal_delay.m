function [x1] = crop_signal_delay(x,sig_output_len,delay)
x1 = x(1:end-delay);
x1 = x1(1:sig_output_len);
end