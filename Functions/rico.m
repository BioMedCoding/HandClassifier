% filtro ricorsivo

function [cMA,cAR]=rico(Rs,B,fc,fs)

% input parameters
%	Rs  (20) attenuazione minima alla frequenza fc in dB
%	B  (2-8) larghezza di banda corrispondente alla attenuazione 0.707
% 	fc (50) frequenza di centro banda
% 	fs  sample frequency
% output parameters
%	cMA  filter coefficients (MA part)
%	cAR  filter coefficients (AR part)

z = 10^(-Rs/20);
T = 1/fs;
b=pi*B*T;
a=b*z;
c1=-2*(1-a)*cos(2*pi*fc*T);
c2=(1-a)^2;
c3=2*(1-b)*cos(2*pi*fc*T);
c4=-(1-b)^2;
cMA=[1 c1 c2];
cAR=[1 -c3 -c4];

end