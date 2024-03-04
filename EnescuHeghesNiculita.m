close all
clear all
clc

% incarcarea setului de date
load('iddata-06.mat');

% load('c1.mat'); % m optim id pred
% load('c2.mat'); % m optim id sim
% load('c3.mat'); % m optim val pred
% load('c4.mat'); % m optim val sim
% 
% load('r1.mat'); % na optim id pred
% load('r2.mat'); % na optim id sim
% load('r3.mat'); % na optim val pred
% load('r4.mat'); % na optim val sim

r1 = 3;
c1 = 3;

r2 = 1;
c2 = 2;

r3 = 1;
c3 = 2;

r4 = 1;
c4 = 2;

% alegerea parametrilor na, nb si a gradului m
na = r2;
nb = na;
m = c2;

% extragerea datelor de identificare si validare
ts = id.Ts; 
u_id = id.u;
y_id = id.y;
u_val = val.u;
y_val = val.y;

N = length(y_id);

%%%%%%%%%%%%%%%%%%%%

% generam toate combinatiile de puteri posibile
puteri_posibile = permutari(na+nb, m);

% eliminand toate combinatiile de puteri care adunate, dau un grad mai mare decat m
capatul_puterilor = zeros(1, na*2);
for i = 1:length(puteri_posibile)
    sum = 0;
    for j = 1:na*2
        sum = sum + puteri_posibile(i, j);
    end
    
    if sum <= m
        capatul_puterilor = [capatul_puterilor; puteri_posibile(i, :)];
    end
end
capatul_puterilor = capatul_puterilor(2:end, :);

%% PREDICTIE Y ID
for i = 1:N
    Xid_pred = zeros(1, 2*na);
    for j = 1:na+nb
        if i-j+na > 0 && j > na
            Xid_pred(j) = u_id(i-j+na);
        end
        if i-j>0 && j <= na
            Xid_pred(j) = -y_id(i-j);
        end
    end
    
    for i1 = 1:nchoosek(m+na+nb, m)
        produs = 1;
        for j1 = 1:na+nb
            produs = produs*Xid_pred(j1)^capatul_puterilor(i1, j1);
            % X1 X2 X3 X4 ...
        end
        phi_id(i, i1) = produs;
    end
end

theta = phi_id\y_id;

predictie_y_id = phi_id * theta;

%EROARE PREDICTIE IDENTIFICARE
MSE_id_pred = 0;

for p=1:N
    MSE_id_pred = MSE_id_pred + (y_id(p)-predictie_y_id(p))*(y_id(p)-predictie_y_id(p));
end

MSE_id_pred = MSE_id_pred/N;

figure;
plot(predictie_y_id);
hold on;
plot(y_id);
title('Predictie pe datele de identificare & MSE=', MSE_id_pred);
legend('y_p_r_e_d', 'y_i_d');


%% PREDICTIE Y VAL
for i = 1:N
    Xval_pred = zeros(1, 2*na);
    for j = 1:na+nb
        if i-j+na > 0 && j > na
            Xval_pred(j) = u_val(i-j+na);
        end
        if i-j>0 && j <= na
            Xval_pred(j) = -y_val(i-j);
        end
    end
    
    for i1 = 1:nchoosek(m+na+nb, m)
        produs = 1;
        for j1 = 1:na+nb
            produs = produs*Xval_pred(j1)^capatul_puterilor(i1, j1);
            % X1 X2 X3 X4 ...
        end
        phi_val(i, i1) = produs;
    end
end

predictie_y_val = phi_val*theta;

%EROARE PREDICTIE VALIDARE
MSE_val_pred = 0;

for p=1:N
    MSE_val_pred = MSE_val_pred + (y_val(p)-predictie_y_val(p))*(y_val(p)-predictie_y_val(p));
end

MSE_val_pred = MSE_val_pred/N;

figure;
plot(predictie_y_val);
hold on;
plot(y_val);
title('predictie pe datele de validare & MSE=', MSE_val_pred);
legend('y_p_r_e_d', 'y_v_a_l');


%% SIMULARE Y ID
simulare_y_id = zeros(1, N); % vectorul pe care il adunam element cu element pt a forma y tilda
Xid_sim = zeros(1, na+nb);
phi_simulare = []; 

for i = 2:N
    for j = 1:na+nb
        if (j<=na)
            if (j>=i)
                Xid_sim(1,j) = 0;
            end
            if (j<i)
                Xid_sim(1,j) = -simulare_y_id(1,i-j);
            end
        end

        if (j>na)
             if (j-na>=i)
                Xid_sim(1,j) = 0;
            end
            if (j-na<i)
                Xid_sim(1,j) = u_id(i-(j-na),1);
            end
        end
    end
    
    for i1 = 1:nchoosek(m+na+nb, m)
        produs = 1;
        for j1 = 1:na+nb
            produs = produs*((Xid_sim(j1))^capatul_puterilor(i1, j1));
            % X1 X2 X3 X4 ... face ridicare la putere
        end
        phi_simulare(i,i1) = produs;
    end
      sum = 0;
        for k=1:nchoosek(m+na+nb,m)
            sum = sum + theta(k,1)*phi_simulare(i, k);
        end
        simulare_y_id(1, i) = sum;
end

%EROARE SIMULARE IDENTIFICARE
MSE_id_sim = 0;

for p=1:N
    MSE_id_sim = MSE_id_sim + (y_id(p)-simulare_y_id(p))*(y_id(p)-simulare_y_id(p));
end

MSE_id_sim = MSE_id_sim/N;

figure;
plot(simulare_y_id);
hold on
plot(y_id);
title('Simulare pe datele de identificare & MSE=', MSE_id_sim);
legend('y_s_i_m', 'y_i_d');


%% SIMULARE Y VAL
simulare_y_val = zeros(1, N); % vectorul pe care il adunam element cu element pt a forma y tilda
Xval_sim = zeros(1, na+nb);
phi_simulare = []; 

for i = 2:N
    for j = 1:na+nb
        if (j<=na)
            if (j>=i)
                Xval_sim(1,j) = 0;
            end
            if (j<i)
                Xval_sim(1,j) = -simulare_y_val(1,i-j);
            end
        end

        if (j>na)
             if (j-na>=i)
                Xval_sim(1,j) = 0;
            end
            if (j-na<i)
                Xval_sim(1,j) = u_val(i-(j-na),1);
            end
        end
    end
    
    for i1 = 1:nchoosek(m+na+nb, m)
        produs = 1;
        for j1 = 1:na+nb
            produs = produs*((Xval_sim(j1))^capatul_puterilor(i1, j1));
            % X1 X2 X3 X4 ... face ridicare la putere
        end
        phi_simulare(i,i1) = produs;
    end
      sum = 0;
        for k=1:nchoosek(m+na+nb,m)
            sum = sum + theta(k,1)*phi_simulare(i, k);
        end
        simulare_y_val(1, i) = sum;
 end

 %EROARE SIMULARE VALIDARE
MSE_val_sim = 0;

for p=1:N
    MSE_val_sim = MSE_val_sim + (y_val(p)-simulare_y_val(p))*(y_val(p)-simulare_y_val(p));
end

MSE_val_sim = MSE_val_sim/N;

figure;
plot(simulare_y_val);
hold on
plot(y_val);
title('Simulare pe datele de validare & MSE=', MSE_val_sim);
legend('y_s_i_m', 'y_v_a_l');


%% functia care genereaza toate combinatiile de numere intre 0 si m
function puteri = permutari(na, m)

    puteri = [];
    indici = zeros(1, na);

    while true
        puteri = [puteri; indici];
        i = na;
        while i > 0 && indici(i) == m
            indici(i) = 0;
            i = i - 1;
        end
        if i == 0
            break;
        end
        indici(i) = indici(i) + 1;
    end
end