function [X,x]=stft_frambld(x,M,L)

% Description:
%   FRAMe BuiLD - divide the signal to (optionally overlapping) frames
% Usage:
%    X=frambld(x,M,L)
% Inputs:
%    x - input signal (can be complex)
%    M - frame size (samples)
%    L - frame step (samples)
% Outputs:
%    X - M on T matrix (T is the number of frames), where
%         each column is one frame

Norig=length(x);
T=floor(Norig/L);
N=L*T+M-L;
x=[x;zeros(N-Norig,1)];
indxsmat=(1:M).'*ones(1,T)+L*ones(M,1)*(0:T-1);
X=x(indxsmat);
