% Michael Zakoworotny
% Plotting parametric results for composite roller printing simulation

clear all; clc; close all;
formatMike();

data = importdata('composite_print_nondim_doe.xlsx');

% Material parameters
% DCPD resin properties
% Thermo-chemical parameters
rhor = 980;
kr =  0.15;
Cpr = 1600.0;
R_ = 8.314;
% Nature values
Hr =  350000;
A_ = 8.55e15;
Er = 110750.0;
n_ = 1.72;
m_ = 0.77;
Ca = 14.48;
alpha_c = 0.41;

% HexTow AS4 Carbon Fiber
rhof = 1790.0;
kf = 6.83;
Cpf = 1129.0;

% Constant process parameters
phi_1 = 0.3;
phi_2 = 0.4;
T0 = 20 + 273.15;
alpha0 = 0.1;

% Homogenized properties at reference fiber volume fraction (phi_2)
phi0 = phi_2;
k_par_o = (1-phi0)*kr + phi0*kf;
k_perp_o = kr + 2*phi0*kr / ((kf+kr)/(kf-kr) - phi0 + (kf-kr)/(kf+kr)*(0.30584*phi0^4 + 0.013363*phi0^8));
rho_Cp_o = (1-phi0)*rhor*Cpr + phi0*rhof*Cpf;

% Variable process parameters
omega_R = data.data(4:end,1);
T_R = data.data(4:end,3) + 273.15; % K
H_gap = data.data(4:end,6)/1000; % m
V_in = data.data(4:end,12)/1000; % m/s
R = data.data(4:end,2)/1000; % m
l_R = data.data(4:end,14)/1000; % m

% Outputs
alpha_max = data.data(4:end,19);
L_front = data.data(4:end,20); % nondimensional front length
Q_in = data.data(4:end,21); % energy per unit thickness (for both rollers)

% Characteristic values
T_max = T0 + rhor*Hr/rho_Cp_o*(1-phi0)*(1-alpha0);
v_c = (A_*R_*k_par_o*T_max^2/rhor/Hr/Er/(1-phi0)/(1-alpha0) * exp(-Er/R_/T_max))^(1/2);
L_c = k_par_o/rho_Cp_o/v_c;
L_front = L_front * L_c; % m

% Reformat data into 3D matrices, (T_R, V_in, H_gap)
num_T = length(unique(T_R(~isnan(T_R))));
num_v = length(unique(V_in(~isnan(V_in))));
num_H = length(unique(H_gap(~isnan(H_gap))));
T_R = reshape(T_R(~isnan(T_R)), num_T, num_v, num_H);
V_in = reshape(V_in(~isnan(V_in)), num_T, num_v, num_H);
H_gap = reshape(H_gap(~isnan(H_gap)), num_T, num_v, num_H);
alpha_max = reshape(alpha_max(~isnan(alpha_max)), num_T, num_v, num_H);
Q_in = abs(reshape(Q_in(~isnan(Q_in)), num_T, num_v, num_H));
l_R = reshape(l_R(~isnan(l_R)), num_T, num_v, num_H);
% Note that L_front still has nan values from no reaction - remove
% corresponding values in other results
L_front = reshape(L_front(~isnan(data.data(4:end,1))), num_T, num_v, num_H);
alpha_max(isnan(L_front)) = nan;
Q_in(isnan(L_front)) = nan;
l_R(isnan(L_front)) = nan;

% Nondimensional process parameters
theta_R = (T_R - T0)/(T_max - T0);
V_star = V_in / v_c;
H_gap_star = H_gap / L_c;
% Nondimensional outputs
L_front_star = L_front / L_c;
Q_in_star = Q_in / k_par_o; % nondimensionalize energy per thickness by reference thermal conductivity

% For parametric plots:
colors = [0 0 0;
          1 0 0;
          1, 0.6445, 0;
          0, 0.3906, 0;
          0, 0, 1;
          0.5, 0, 0.5];
symbs = {'s','^','v','o','*','d'};




%%
% Plot the maximum degree of cure, front location and energy input Q_in as 
% functions of V and T_R, for a given value of Hgap

H_plot_list = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]; % values of Hgap to plot for
for i = 1:length(H_plot_list)
    val_H_plot = H_plot_list(i);
    ind_H_plot = find(H_gap(1,1,:) == val_H_plot);

    % Contour plot of alpha_max vs (V_in, T_R) for a given value of V_in
    fig = figure; box on; hold on;
    [contour_M, contour_c] = contourf(squeeze(T_R(:,:,ind_H_plot))-273.15, squeeze(V_in(:,:,ind_H_plot))*1000, squeeze(alpha_max(:,:,ind_H_plot)));%, [0.6:0.025:0.95]);
    colormap(pmkmp(256,'cubicl')); 
    scatter(reshape(T_R(:,:,ind_H_plot),1,[])-273.15, reshape(V_in(:,:,ind_H_plot),1,[])*1000, 20, 'sk', 'filled'); % add data points
    cb = colorbar; title(cb, '$\mathrm{\alpha_{max}}$','interpreter','latex','fontsize',14); %cb.Label.String = '$\alpha_{max}$';
    clabel(contour_M, contour_c, [0.65:0.05:0.95], 'fontsize',14, 'interpreter','latex'); %,'Rotation',0 (can only use w/o contour_c)
    xlabel('$\mathrm{T_R}$~[${}^{\circ}$C]');
    ylabel('$\textrm{V}_{in}$ [mm/s]');
    title(['$\textrm{H}_{gap}$ = ', num2str(val_H_plot*1000,2), ' mm']);
%     saveas(fig, ['plots/contour_alphaMax_Vin_Tr_Hgap',replace(num2str(val_H_plot*1000),'.','p'),'.png']);

    % Contour plot of Q_in vs (V_in, T_R) for a given value of H_gap
    fig = figure; box on; hold on;
    [contour_M, contour_c] = contourf(squeeze(T_R(:,:,ind_H_plot))-273.15, squeeze(V_in(:,:,ind_H_plot))*1000, squeeze(Q_in(:,:,ind_H_plot))/1e3);%, [0.65:0.025:0.95]);
    colormap(pmkmp(256,'cubicl')); 
    scatter(reshape(T_R(:,:,ind_H_plot),1,[])-273.15, reshape(V_in(:,:,ind_H_plot),1,[])*1000, 20, 'sk', 'filled'); % add data points
    cb = colorbar; title(cb, '$\textrm{Q}_{in}$ [kW/m]','interpreter','latex','fontsize',14); %cb.Label.String = '$\alpha_{max}$';
    % clabel(contour_M, contour_c, [0.65:0.05:0.95], 'fontsize',14, 'interpreter','latex'); %,'Rotation',0 (can only use w/o contour_c)
    xlabel('$\mathrm{T_R}$~[${}^{\circ}$C]');
    ylabel('$\textrm{V}_{in}$ [mm]');
    title(['$\textrm{H}_{gap}$ = ', num2str(val_H_plot*1000,2), ' mm']);
%     saveas(fig, ['plots/contour_Qin_Vin_Tr_Hgap',replace(num2str(val_H_plot*1000),'.','p'),'.png']);
    
    % Curves for L_front vs (V_in, T_R) for a given value of Hgap
    fig = figure; box on; hold on;
    colororder(colors);
    scrn = get(0,'ScreenSize');
    set(gcf,'Position',[0.1*scrn(3), 0.1*scrn(4), 0.4*scrn(3), 0.42*scrn(4)]);
    % colororder()
    plts = [];
    for j = 2:2:size(V_in,2)
        plt_i = plot(squeeze(T_R(:,j,ind_H_plot))-273.15, squeeze(L_front(:,j,ind_H_plot))./l_R(:,j,ind_H_plot), ['-',symbs{j/2}],'LineWidth',1.5,'DisplayName',['$\textrm{V}_{in} = $',num2str(V_in(1,j,1)*1000),'mm/s']);
        plts = [plts, plt_i];
    end
    plot(xlim, [0,0],'--k','LineWidth',1.5);
    xlabel('$\textrm{T}_R$~[${}^{\circ}$C]'); % xlabel('$\omega_R$ [rad/s]'); 
    ylabel('$\mathrm{L_{front}/\ell_R}$ ');
    title(['$\textrm{H}_{gap}$ = ', num2str(val_H_plot*1000,2), ' mm']);
    legend(plts, 'location','eo');
%     saveas(fig, ['plots/plot_Lfront_Tr_Vin_Hgap',replace(num2str(val_H_plot*1000),'.','p'),'.png']);
    
end



