function A = AofT(par, t)

% Function A(t)
% Inputs: 
%   par: parameters
%   t: maturities
% Outputs: 
%   A: A(t)

% Parameters
kappa    = par(1);
gamma    = par(2);
mu     = par(3);
sigmachi = par(4);
sigmaxi  = par(5);
rho      = par(6);
lambdachi = par(7);
lambdaxi = par(8);

A = -(lambdachi/kappa)*(1-exp(-kappa*t)) + (mu - lambdaxi)/gamma*(1-exp(-gamma*t)) + ...
        0.5*((1-exp(-2*kappa*t))*sigmachi^2/2/kappa + (1-exp(-2*gamma*t))*sigmaxi^2/2/gamma + 2*(1-exp(-(kappa+gamma)*t))*sigmachi*sigmaxi*rho/(kappa+gamma));
