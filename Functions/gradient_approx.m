function gradient = gradient_approx(par, delta, yt, mats, dt)

% Calculate the first order derivative at special point "par", with
% increment "delta"
% Central difference 

% Inputs: 
%   par: estimates of parameters
%   delta: increments, a vector with same length of par
%   yt: data
%   mats: maturities
%   dt: delta t
% Outputs: 
%   gradient: approximated gradient vector

n_par = length(par);

if length(delta) ~= n_par
    error('The length of delta must be equal to the number of parameters');
end

delta_mat = diag(delta);
gradient = zeros(1, n_par); 

for i = 1: n_par
    del = delta_mat(i, :);
    [ll1, Q1, af, at, as, ft] = KalmanFilter(par - del, yt, mats, dt, true, false);
    [ll2, Q2, af, at, as, ft] = KalmanFilter(par + del, yt, mats, dt, true, false);
   
    gradient(i) = (Q2 - Q1) / ( 2*delta(i) );
end
