function [logL, Q, table_at_filter, table_at_prediction, table_at_smoother, ft] = KalmanFilter(par, yt, T, delivery_time, dt, smoothing, seasonality)

% Two dimensional Kalman Filter & Smoother. This function is modified from twodimoukf_v5. 

% Model: 
%   y_t = d_t + F_t' x_t + f_t + v_t, v_t ~ N(0, V), observation equation
%   x_t = c + G x_{t-1} + w_t, w_t ~ N(0, W), state equation
%   f_t = b*t + beta*cos(2*pi*t*dt) + eta*sin(2*pi*t*dt), seasonal effect
% Inputs: 
%   par: a vector of parameters
%   yt: the logarihm of futures prices
%   T: maturities
%   date: a vector of date, which is necessary if seasonality is "Constant"
%   dt: delta t
%   smoothing: a boolean variable indicate if Kalman Smoothing is required
%   seasonality: "Constant" or "None"
% Outputs: 
%   logL: the negative log likelihood 
%   Q: used to calculate the asymptotic varainces of estimates. See
%       Koopman (1992) and Durbin & Koopman (2001) for details. 
%   table_at_filter: a nT*2 matrix gives the filtered values of state
%       variables. 
%   table_at_prediction: a (nT+1)*2 matrix gives the predicted values of
%       state variables. 
%   table_at_smoother: a nT*2 matrix gives the smoothed values of state
%       variables. The algorithm of Kalman Smoother is given by Bierman
%       (1973) and De Jong (1989). 

% nT: number of observations
% n: number of contracts
[nT, n] = size(yt); 

table_at_filter = zeros(nT, 2); % a_t|t
table_Pt_filter = zeros(2, 2, nT); % P_t|t
table_at_prediction = zeros(nT+1, 2); % a_t|t-1
table_Pt_prediction = zeros(2, 2, nT+1); % P_t|t-1
table_at_smoother = zeros(nT, 2); % a_t|s, s > t
table_Pt_smoother = zeros(2, 2, nT); % P_t|s, s > t

table_et = zeros(nT, n); % e_t
table_L = zeros(n, n, nT); % L_t|t-1
table_invL = zeros(n, n, nT); % inverse of L_t|t-1
table_K = zeros(2, n, nT); % Kalman gain matrix
table_D = zeros(nT, n); % d_t
table_F = zeros(2, n, nT); % F_t

% Tables used for gradient
table_en = zeros(nT, n); % et|n
table_wn = zeros(nT, 2); % Smoothed estimates of wt
table_var_en = zeros(n, n, nT); % variance of en
table_var_wn = zeros(2, 2, nT); % variance of wn
tr1 = zeros(1, nT);
tr2 = zeros(1, nT);

% Seasonal component
if seasonality == "Constant"
    par_seasonal = par(end-11: end);
    ft = zeros(nT, n);
    for i = 1: nT
        ft(i, :) = par_seasonal*(month(delivery_time(i)') == 1: 12)';
    end
    par = par(1: end-12);    
%elseif seasonality == "Trigonometric"
%    alpha = par(end-2);
%    beta = par(end-1);
%    eta = par(end);
%    par = par(1: end-3);
    %f_t = b*t + beta*cos(2*pi*t*dt) + eta*sin(2*pi*t*dt)
%    seasonal_time = 1: nT; % t
%    ft = alpha*seasonal_time + beta*cos(2*pi*seasonal_time*dt) + eta*sin(2*pi*seasonal_time*dt); 
elseif seasonality == "None"
    ft = 0;
else
    error("The seasonal component must be 'Constant', 'Trigonometric' or 'None'. ");
end

% Parameters
kappa = par(1);
gamma = par(2);
muxi = par(3);
sigmachi = par(4);
sigmaxi = par(5);
rho = par(6);
lambdachi = par(7);
lambdaxi = par(8);

if length( par(9: end) ) ~= n
    error("The number of standard errors doesn't match the number of contracts. ");
else
    V = diag( par(9: end).^2 );
end

% Initial a and P
at_prediction = [ -lambdachi / kappa ; (muxi - lambdaxi) / gamma ]; % a_1|0
Pt_prediction = [ sigmachi^2 / (2*kappa), sigmachi*sigmaxi*rho / (kappa + gamma); 
    sigmachi*sigmaxi*rho / (kappa + gamma), sigmaxi^2 / (2*gamma)]; % P_1|0 

table_at_prediction(1, :) = at_prediction';
table_Pt_prediction(:, :, 1) = Pt_prediction; 

% Parameters for state equation
c  = [ -lambdachi/kappa*(1-exp(-kappa*dt)) ; (muxi - lambdaxi)/gamma*(1-exp(-gamma*dt))];
G  = [exp(-kappa*dt), 0; 0, exp(-gamma*dt)];
sigmastate1 = (1-exp(-2*kappa*dt))/(2*kappa)*sigmachi^2; 
sigmastate2 = (1-exp(-2*gamma*dt))/(2*gamma)*sigmaxi^2;
covstate = (1-exp(-(kappa+gamma)*dt))/(kappa+gamma)*sigmachi*sigmaxi*rho;
W = [sigmastate1, covstate; covstate, sigmastate2]; % covariance matrix of w_t

detL = zeros(nT, 1); % the determinant of L 
eLe = zeros(nT, 1); % e'*inv(L)*e in log-likelihood function
I = eye(2);

n1 = 0; 

% Kalman Filter
for j = 1:nT
    yt1 = yt(j, ~isnan(yt(j,:)));
    n1 = n1 + length(yt1); % add number of obs. for this time
    
    D  = AofT(par, T(j,:))' + ft(j); % d_t + f_t
    F = [exp(-kappa*T(j,:)); exp(-gamma*T(j,:))]'; % F_t
    
    y_pred = D + F * at_prediction; % ytilde_t|t-1 = d_t + F_t a_t|t-1
    L = F * Pt_prediction * F' + V; % L
    
    %check if matrix is semi positive definite
    % eigenvalues of (Py+Py')/2 should be positive
    if  sum(sum(diag(eig((L+L')/2)<0)))>0
        disp('matrix is not semi positive definite');
    end
    
    et = yt1' - y_pred; %e_t = y_t - ytilde_t|t-1
    
    % store values for likelihood
    detL(j, :) = det(L);
    invL = L\eye(n); % inverse of L 
    eLe(j)  = et' * invL * et; 
    K  = Pt_prediction * F' * invL; % Kalman gain matrix: K_t
    
    % Filter
    at_filter = at_prediction + K*et; % a_t
    Pt_filter = (I - K*F)*Pt_prediction; % P_t
    
    % Prediction
    at_prediction  = c + G * at_filter; % a_t+1|t 
    Pt_prediction = G*Pt_filter*G' + W; % P_t+1|t
    
    % Update tables
    table_at_filter(j, :) = at_filter';
    table_Pt_filter(:, :, j) = Pt_filter;
    table_at_prediction(j+1, :) = at_prediction';
    table_Pt_prediction(:, :, j+1) = Pt_prediction;
    
    table_et(j, :) = et';
    table_L(:, :, j) = L;
    table_invL(:, :, j) = invL; 
    table_K(:, :, j) = K;
    table_D(j, :) = D';
    table_F(:, :, j) = F';
end

% Kalman Smoother
if smoothing
    for t = nT: -1: 1
        yt1 = yt(j,~isnan(yt(t,:)));
        F = table_F(:, :, t)';
        D = table_D(t, :)';
        K = table_K(:, :, t);
        invL = table_invL(:, :, t);
        et = table_et(t, :)';
        
        at_prediction = table_at_prediction(t, :)';
        Pt_prediction = table_Pt_prediction(:, :, t);
        
        if t == nT
            rt = [0; 0]; 
            Rt = [0, 0; 0, 0];
        end
        
        rt = F' * invL * et + (G - G * K * F)' * rt; % 2 * 1 matrix 
        Rt = F' * invL * F + (G - G * K * F)' * Rt * (G - G * K *F); % 2 * 2 matrix
     
        at_smoother = at_prediction + Pt_prediction * rt; % a_t|n
        Pt_smoother = Pt_prediction - Pt_prediction * Rt * Pt_prediction; %P_t|n
        
        en = V * invL * et - V * K' * G' * rt; % n * 1 matrix
        wn = W * rt; % 2 * 1 matrix
        
        var_en = V - V * (invL + K' * G' * Rt * G * K) * V; % variance of e_n 
        var_wn = W - W * Rt * W; % variance of w_n
        
        % Update tables
        table_at_smoother(t, :) = at_smoother;
        table_Pt_smoother(:, :, t) = Pt_smoother;
        table_en(t, :) = en';
        table_wn(t, :) = wn';
        table_var_en(:, :, t) = var_en;
        table_var_wn(:, :, t) = var_wn;
        tr1(t) = trace( (en * en' + var_en) * inv(V) );
        tr2(t) = trace( (wn * wn' + var_wn) * inv(W) );
    end
    
end

if smoothing
    Q1 = nT * log(det(V));
    Q2 = nT * log(det(W));
    Q3 = sum(tr1);
    Q4 = sum(tr2);
    Q = -0.5 * (Q1 + Q2 + Q3 + Q4);
else
    Q = 0;
end

%calculate loglikelihood
logL = -.5*n1*log(2*pi) - .5*sum(log(detL)) - .5*sum(eLe);
logL = -logL;

