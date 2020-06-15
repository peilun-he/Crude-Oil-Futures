clear all;

% Set search path
addpath(genpath(pwd));

% Parameters 
kappa = 1.5;
gamma = 1;
mu = -2;
sigma_chi = 0.5;
sigma_xi = 0.3;
rho = -0.7;
s1 = 0.03;
par_seasonal = [0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.2, -0.1];

% Setups 
n_obs = 1000; % number of observations 
M = 1000; % number of iterations to calculate asymptotic variance
h = 0.001; % increments to calculate asymptotic variance
n_contract = 13; % number of futures contracts 

par_org = [kappa, gamma, mu, sigma_chi, sigma_xi, rho, 0, 0, repelem(s1, n_contract), par_seasonal]; % original parameters 

monthdays = 30; % days per month
yeardays = 360; % days per year 
dt = 1/yeardays; % delta t
seed = 123; 
x0 = [2.5, 1.2];
start_date = datetime("2000-01-01", "InputFormat", "yyyy-MM-dd");

n_para = 7; % number of different parameters to be estimated 
% In this example, we don't estimate lambdachi and lambdaxi, and only estimate one 's' in matrix V.
% We have parameters: kappa, gamma, mu, sigmachi, sigmaxi, rho, s1
% So n_para = 7. This number must be handled careful. 
n_grid = 3; % number of grid points

% Bounds and constraints 
parL = [10^(-5), 10^(-5),   -5,  0.01,  0.01,  -0.9999, 0, 0, repelem(10^(-5), n_contract), 0, repelem(-2, 11)]; % lower bound
parU = [      3,       3,    4,     3,     3,   0.9999, 0, 0, repelem(1, n_contract),       0, repelem(2, 11) ]; % upper bound
A = [-1, 1, 0, 0, 0, 0, 0, 0, repelem(0, n_contract), repelem(0, 12)]; % constraint: kappa >= gamma 
b = 0;
Aeq = []; % Equal constraints: Aeq*x=beq
beq = []; 

for i = 1: n_contract-1
    Aeq = [Aeq; repelem(0, 7+i), 1, -1, repelem(0, n_contract-1-i), repelem(0, 12)];
    beq = [beq; 0];
end

% Simulate futures prices and maturities 
[yt, mats, xt, ft, date, delivery_time] = SimulateYtMats(par_org, x0, n_obs, n_contract, monthdays, yeardays, "Constant", start_date, seed); 

% Parameter estimation 
par0 = [2, 2, 0, 1, 1, 0.5, 0, 0, repelem(0.1, n_contract), 0, repelem(0.1, 11)]; 
options = optimset('TolFun', 1e-06, 'TolX', 1e-06, 'MaxIter', 2000, 'MaxFunEvals', 4000);
[par_est, nll, exitflag] = fmincon(@KalmanFilter, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, date, dt, false, "Constant");

% Asymptotic variance
%method = 2;
%seed = 1234;
%[asyVar, message] = AsymptoticVariance(par_est, x0, yt, M, h, method, monthdays, yeardays, seed);

% Grid search 
mid = (parL + parU) / 2;
MP1 = (mid + parL) / 2; % midpoint of lower bound and mid
MP2 = (mid + parU) / 2; % midpoint of upper bound and mid

if n_grid == 2
    grid = [MP1; MP2]';
elseif n_grid == 3
    grid = [MP1; mid; MP2]';
else 
    disp('The number of grid points is wrong. ');
end

est = zeros(n_grid^n_para, length(par_org)+1); % estimates of parameters and NLL at each point
init = []; % initial values

for ka = grid(1, :)
    for ga = grid(2, :)
        for m = grid(3, :)
            for sc = grid(4, :)
                for sx = grid(5, :)
                    for rh = grid(6, :)
                            init = [init; ka, ga, m, sc, sx, rh, 0, 0, repelem(0.1, n_contract), repelem(0, 12)];
                    end
                end
            end
        end
    end
end

for i = 1: n_grid^n_para
    par0 =  init(i, :);
    options = optimset('TolFun',1e-06,'TolX',1e-06,'MaxIter',1000,'MaxFunEvals',2000);
    [par, fval, exitflag] = fmincon(@KalmanFilter, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, dt, false, false);
    est(i, :) = [par, fval];
end

index = ( est(:, end) == min(est(:, end)) );
best_est = est(index, :);
best_init = init(index, :);

% Asymptotic variances
method = 2;
seed = 1234;
[asyVar, message] = AsymptoticVariance(best_est(1: end-1), x0, yt, M, h, method, monthdays, yeardays, seed);

% Figures
[ll, Q, af, ap, as, ft_est] = KalmanFilter(par_est, yt, mats, date, dt, true, "Constant");
f_filter = zeros(n_obs, n_contract);

for i = 1: n_obs
    D = AofT(par_est, mats(i, :))' + ft_est(i);
    F = [exp(-par_est(1) * mats(i, :)); exp(-par_est(2) * mats(i, :))]';
    y_filter(i, :) = D + F*af(i, :)';
end

date = 1: n_obs;

figure;
hold on;
plot(date, ft, 'r');
plot(date, yt(:, 1), 'k', 'LineWidth', 1);
plot(date, xt(:, 1), 'b');
plot(date, xt(:, 2), 'g');
hold off;
xlabel("Date");
ylabel("Price");
legend("Seasonal effect", "1st contract", "\chi", "\xi");

figure;
subplot(2, 2, 1);
plot(date, yt(:, 1), 'k', date, y_filter(:, 1), 'r');
xlabel("Date");
ylabel("Price");
legend("Simulated 1st contract", "Estimated 1st contract");

subplot(2, 2, 2);  
plot(date, xt(:, 1), 'k', date, af(:, 1), 'r');
xlabel("Date");
ylabel("Price");
legend("Simulated \chi", "Estimated \chi");

subplot(2, 2, 3);  
plot(date, xt(:, 2), 'k', date, af(:, 2), 'r');
xlabel("Date");
ylabel("Price");
legend("Simulated \xi", "Estimated \xi");

subplot(2, 2, 4);  
plot(date, ft, 'k', date, ft_est, 'r');
xlabel("Date");
ylabel("Price");
legend("Simulated seasonal effect", "Estimated seasonal effect");

% Power spectral density
X = yt(:, 1);
figure;
plot(1: 1000, X);
%Y = fft(X);
%figure;
%plot(1: 1000, Y);
pxx = pwelch(yt, 100, 90);
figure
plot(1: 129, pxx)






