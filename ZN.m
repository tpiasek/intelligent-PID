init;
model_name = 'schemat.slx';

tau_vec  = [0.0, 0.1, 0.2, 0.3, 0.5];    % opóźnienia (delays)
r_vec    = [0.3, 0.6, 1.0, 1.5, 2.0];    % wartości zadane
umax_vec = [1.5, 3.0, 100.0];            % limity nasycenia (100 = brak)

results_table = [];
scenario_id = 1;

fprintf('Uruchamianie symulacji\n');

for t_idx = 1:length(tau_vec)
    for r_idx = 1:length(r_vec)
        for u_idx = 1:length(umax_vec) %uruchamiamy model w Simulinku na czas 30 sekund z aktualnie ustawionymi parametrami tau, r, Umax - wszystkie kombinacje
            
            tau  = tau_vec(t_idx);
            r    = r_vec(r_idx);
            Umax = umax_vec(u_idx);
            
            simOut = sim(model_name, 'StopTime', '30');
            
            t_sim  = simOut.sim_y.Time;
            r_sim  = r * ones(size(t_sim));
            u_pid  = simOut.sim_u_pid.Data;
            u_real = simOut.sim_u_real.Data;
            y_out  = simOut.sim_y.Data;
            
            sc_id  = scenario_id * ones(size(t_sim));
            tau_c  = tau * ones(size(t_sim));
            umax_c = Umax * ones(size(t_sim));
            
            temp_data = [t_sim(:), r_sim(:), u_pid(:), u_real(:), y_out(:), sc_id(:), tau_c(:), umax_c(:)];
            results_table = [results_table; temp_data];
            
            scenario_id = scenario_id + 1;
        end
    end
end

data_headers = {'time', 'ref_r', 'u_pid', 'u_real', 'y_out', 'scenario_id', 'tau_s', 'umax'}; %eksport do csv
T_out = array2table(results_table, 'VariableNames', data_headers);
writetable(T_out, 'baseline_data.csv', 'Delimiter', ';');
fprintf('Zapisano baseline_data.csv (%d scenariuszy).\n', scenario_id - 1);