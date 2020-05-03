function [asyVar, message] = AsymptoticVariance(par, x0, yt, M, increment, method, monthdays, yeardays, s)

% Calculate asymptotic variances of parameters. If any variances are
% negative, they will be replaced by their absolute values. 
% This function is modified from asymptoticVaraince.

% Inputs: 
%   par: estimates of parameters
%   x0: initial state vector
%   yt: data
%   M: number of iterations
%   method: 1 -> numerical second order derivative
%           2 -> product of gradient
%   increment: increment to calculate the numerical gradient / second order derivative 
%   monthdays: days per month
%   yeardays: days per year
%   s: seed for random values
% Outputs: 
%   asyVar: asymptotic variance matrix
%   message: 0 -> normal
%            1 -> some variances are negative

n_par = length(par);
[nobs, ncontract] = size(yt);
dt = 1 / yeardays;

hess = zeros(n_par, n_par, M); % hessian matrix

if method == 1
    % f''(x) = ( f(x+delta1+delta2) - f(x+delta1-delta2) - f(x-delta1+delta2) + f(x-delta1-delta2) ) / (4*delta1*delta2)
    
    % increments used for method 1
    incre1 = repelem(increment, n_par); % increment vector
    incre2 = repelem(increment, n_par); % increment vector 
    incre1_mat = diag(incre1); % matrix
    incre2_mat = diag(incre2); % matrix
    
    for m = 1: M
        [yt, mats] = SimulateYtMats(par, x0, nobs, ncontract, monthdays, yeardays, false, s+m); % simulate yt and maturities
                
        for i = 1: n_par
            for j = 1: n_par
                [ll1, Q, Q_vec, af, at, as] = KalmanFilter(par + incre1_mat(i, :) + incre2_mat(j, :), yt, mats, dt, true);
                [ll2, Q, Q_vec, af, at, as] = KalmanFilter(par + incre1_mat(i, :) - incre2_mat(j, :), yt, mats, dt, true);
                [ll3, Q, Q_vec, af, at, as] = KalmanFilter(par - incre1_mat(i, :) + incre2_mat(j, :), yt, mats, dt, true);
                [ll4, Q, Q_vec, af, at, as] = KalmanFilter(par - incre1_mat(i, :) - incre2_mat(j, :), yt, mats, dt, true);
                hess(i, j, m) = (ll1 - ll2 - ll3 + ll4) / (4 * incre1(i) * incre2(j));
            end
        end    
    end
    
elseif method == 2    
    incre = repelem(increment, n_par); % increment used for method 2
    
    for m = 1: M
        [yt, mats] = SimulateYtMats(par, x0, nobs, ncontract, monthdays, yeardays, false, s+m); % simulate yt and maturities
        
        grad = gradient_approx(par, incre, yt, mats, dt);
        hess(:, :, m) = grad' * grad; 
    end
    
else 
    error('Method must be 1 or 2.');
end

fim = mean(hess, 3); % Fisher information matrix
asyVar = inv(fim); % asymptotic variance

message = 0;

if any( diag(asyVar) < 0 )
    message = 1;
    [V1, D1] = eig(asyVar);
    for i = 1: n_par
        if D1(i, i) <=0 
            D1(i, i) = -D1(i, i); % replace negative variance by its absolute value
        end
    end
    asyVar = V1 * D1 * inv(V1);
end



    
    