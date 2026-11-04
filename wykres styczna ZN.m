init;
model_name = 'schemat.slx';

% Odp skokowa G_lin(s) ze styczną ZN

figure('Name', 'Identyfikacja Z-N');
plot(t_step, y_step, 'b', 'LineWidth', 2); hold on;
t_tangent = max(0, L):0.1:(L + Tz + 1);
y_tangent = max_slope * (t_tangent - t_przeg) + y_przeg;
plot(t_tangent, y_tangent, 'r--', 'LineWidth', 1.5);
plot(t_przeg, y_przeg, 'ro', 'MarkerFaceColor', 'r');
grid on; xlim([0 15]); ylim([0 2.2]);
title('Odpowiedź skokowa zlinearyzowanego obiektu G_{lin}(s)');
xlabel('Czas [s]'); ylabel('y(t)');
legend('G_{lin}(s)', 'Styczna', 'Punkt przegięcia', 'Location', 'SE');