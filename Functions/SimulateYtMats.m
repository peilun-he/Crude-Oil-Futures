function [yt, mats, x, ft, date, delivery_time] = SimulateYtMats(par, X0, nobs, ncontracts, monthdays, yeardays, seasonality, start_date, s)

% Simulate data and maturities. This fnction is modified from simulateYtMats_v3.

% Inputs: 
%   par: parameters
%   X0: initial value for state variable x_t
%   nobs: the number of observations
%   ncontracts: the number of contracts
%   monthdays: how many days in a month
%   yeardays: how many days in a year
%   seasonality: "Constant" or "None"
%   start_date: the beginning of date
%   s: seed for random values
% Outputs: 
%   yt: logarithm of futures prices
%   mats: maturities
%   xt: state vector
%   ft: seasonal effect
%   date: a vector of date

nyears = nobs / yeardays; % number of years
dt1 = 1 / yeardays; % delta_t

% Simulate seasonal effect
if seasonality == "Constant"
    date = start_date: start_date+nobs-1;
    date = date(:);
    date_mat = (month(date) == 1: 12);
    par_seasonal = par(end-11: end);
    ft = date_mat * par_seasonal';
    par = par(1: end-12);
%elseif seasonality == "Trigonometric"
    % ft = b*t + beta*cos(2*pi*t*dt1) + eta*sin(2*pi*t*dt1)
%    alpha = par(end-2);
%    beta = par(end-1);
%    eta = par(end);
%    par = par(1: end-3);
%    seasonal_time = 1: nobs; % t
%    ft = alpha*seasonal_time + beta*cos(2*pi*seasonal_time*dt1) + eta*sin(2*pi*seasonal_time*dt1); 
%    date = [];
elseif seasonality == "None"
    ft = repelem(0, nobs);
    date = [];
else
    error("The seasonal component must be 'Constant', 'Trigonometric' or 'None'. ");
end

% Parameters
kappa     = par(1);
gamma     = par(2);
mu        = par(3);
sigmachi  = par(4);
sigmaxi   = par(5);
rho       = par(6);
lambdachi = par(7);
lambdaxi = par(8);

if length( par(9: end) ) ~= ncontracts
    error("The number of srandard errors doesn't match the number of contracts. ");
else
    V = diag( par(9: end).^2 );
end

% Initial values for chi and xi
chi0      = X0(1);
xi0       = X0(2);

% fix random seed
if s>0
    rng(s);
    disp('fixed random seed');
end

% Random noise 
if rho==1
    Z1 = randn(nobs+1, 1);
    Z  = [Z1, Z1];
else
    Z = mvnrnd([0, 0], [1, rho; rho, 1], nobs+1); % multivariate normal random numbers
end

% Simulate chi and xi
a1 = exp( -kappa * dt1);
a2 = exp( -gamma * dt1);
b1 = -(lambdachi/kappa)*(1-a1);
b2 = ((mu-lambdaxi) / gamma) * ( 1 - a2 );
c1 = sigmachi * sqrt(( 1 - a1^2 ) / ( 2*kappa ));
c2 = sigmaxi * sqrt(( 1 - a2^2 ) / ( 2*gamma ));

chi(1) = chi0;
xi(1) = xi0;

for j=1: nobs+1
    chi(j+1) = a1 * chi(j) + b1 + c1 * Z(j,1); % chi
    xi(j+1)  = a2 * xi(j)  + b2 + c2 * Z(j,2); % xi
end

state = [chi;xi]';

x = state(2: (nobs+1), :); % state variable x_t

% Simulate yt
T = [monthdays: monthdays: ncontracts*monthdays]; 
T = T + 1;
mats  = zeros(nobs, length(T));
logfT = zeros(nobs, length(T));
v1store = zeros(nobs, length(T));

for j=1:nobs
   v1 = mvnrnd(repelem(0, ncontracts), V);
   v1store(j,:)=v1; % error for measurement eq.
   if mod(j-1,monthdays) == 0 & j ~= 1
       T = T + monthdays;
   end
   
   mats(j, :) = (T - j);
   
   if j == 1
       delivery_time = date(j) + mats(j, :);
   else
       delivery_time = [delivery_time; date(j) + mats(j, :)];
   end
   mats(j, :) = mats(j, :) ./ yeardays; % maturities
   logfT(j,:) = x(j, :) * [exp(-kappa*mats(j,:)); exp(-gamma*mats(j, :))] + AofT(par, mats(j,:)) + ft(j) + v1;
end

yt  = logfT;


