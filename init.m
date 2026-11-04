clear all;
close all;

%% opis

% Zbiornik 1 - Prędkość wypływu ze zbiornika zależy od słupa cieczy, co
% powoduje, że zmiana poziomu nie jest wprost proporcjonalna do dopływu.
% Mamy pierwiastek w opisie.

% Zbiornik 2 - ma liniową dynamikę, dlatego w Simulinku jest opisany
% bloczkiem FCN (transmitancji I rzędu)
%%
% parametry
K   = 2.0;       % wzmocnienie
Ta  = 3.0;       % stała czasowa 1 dla zbiornika
Tb  = 2.0;       % stała czasowa 2 dla zbiornika
y0  = 1.0;       % nominalny punkt pracy
Ts  = 0.05;      % krok próbkowania regulatora
tau = 0.0;       % opóźnienie

Umin = 0.0;      % dolny limit saturacja
Umax = 3.0;      % gorny limit saturacja

T1 = Ta + Tb;    % T1 = 5.0 s
T2 = Ta * Tb;    % T2 = 6.0 s^2

% TRANSMITANCJA ZLINEARYZOWANA G_lin(s)
s = tf('s');
G_lin = K / (T2*s^2 + T1*s + 1);

% nastawy Z-N
[y_step, t_step] = step(G_lin, 0:0.005:20); %odpowiedź zlinearyzowanego układu G_lin(s) na skok u=1 w czasie od 0 do 20 s z krokiem 0,005s
dy = diff(y_step) ./ diff(t_step); %pochodna numeryczna, czyli prędkość wzrostu sygnału wyjściowego
[max_slope, idx] = max(dy); % znajduje punkt przegięcia

t_przeg = t_step(idx);
y_przeg = y_step(idx);

L  = t_przeg - y_przeg / max_slope; % Zastępcze opóźnienie
Tz = K / max_slope;                 % Zastępcza stała czasowa

Kp = (1.2 * Tz) / (K * L); % nastawy pid z ZN wzory bazowe
Ti = 2.0 * L;
Td = 0.5 * L;

Ki = Kp / Ti; % paraller pid 
Kd = Kp * Td;

fprintf('NASTAWY PID - ZN\n');
fprintf('Kp = %.4f | Ki = %.4f | Kd = %.4f\n\n', Kp, Ki, Kd);

% weryfikacja petli otwartej 
u_test = 1.8; % u0 (1.0) + delta_u (0.8)
y_ust_teor = ((K * u_test)^2) / (4 * y0);
fprintf('TEST PĘTLI OTWARTEJ\n');
fprintf('Dla u = %.2f oczekiwane y_ust = %.3f (powinno wynosić 3.240)\n\n', u_test, y_ust_teor);