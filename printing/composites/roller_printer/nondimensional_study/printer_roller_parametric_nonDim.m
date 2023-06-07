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

% Nondimensional process parameters
theta_R = data.data(4:end,15);
V_star = data.data(4:end,16);
H_gap_star = data.data(4:end,17);

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

% For parametric plots:
colors = [0 0 0;
          1 0 0;
          1, 0.6445, 0;
          0, 0.3906, 0;
          0, 0, 1;
          0.5, 0, 0.5];
symbs = {'s','^','v','o','*','d'};

% Contour plot of alpha_max vs (H_gap, T_R) for a given value of V_in
v_plot_list = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3];
for i = 1:length(v_plot_list)
    val_V_plot = v_plot_list(i);
    ind_V_plot = find(V_in(1,:,1) == val_V_plot);

    fig = figure; box on; hold on;
    [contour_M, contour_c] = contourf(squeeze(T_R(:,ind_V_plot,:))-273.15, squeeze(H_gap(:,ind_V_plot,:))*1000, squeeze(alpha_max(:,ind_V_plot,:)));%, [0.6:0.025:0.95]);
    colormap(pmkmp(256,'cubicl')); 
    scatter(reshape(T_R(:,ind_V_plot,:),1,[])-273.15, reshape(H_gap(:,ind_V_plot,:),1,[])*1000, 20, 'sk', 'filled'); % add data points
    cb = colorbar; title(cb, '$\mathrm{\alpha_{max}}$','interpreter','latex','fontsize',14); %cb.Label.String = '$\alpha_{max}$';
    clabel(contour_M, contour_c, [0.65:0.05:0.95], 'fontsize',14, 'interpreter','latex'); %,'Rotation',0 (can only use w/o contour_c)
    xlabel('$\mathrm{T_R}$~[${}^{\circ}$C]');
    ylabel('$\textrm{H}_{gap}$ [mm]');
    title(['$\textrm{V}_{in}$ = ', num2str(val_V_plot*1000,2), ' mm/s']);
    saveas(fig, ['contour_alphaMax_Hgap_Tr_Vin',replace(num2str(val_V_plot*1000),'.','p'),'.png']);

    % Contour plot of Q_in vs (H_gap, T_R) for a given value of V_in
    fig = figure; box on; hold on;
    [contour_M, contour_c] = contourf(squeeze(T_R(:,ind_V_plot,:))-273.15, squeeze(H_gap(:,ind_V_plot,:))*1000, squeeze(Q_in(:,ind_V_plot,:))/1e3);%, [0.65:0.025:0.95]);
    colormap(pmkmp(256,'cubicl')); 
    scatter(reshape(T_R(:,ind_V_plot,:),1,[])-273.15, reshape(H_gap(:,ind_V_plot,:),1,[])*1000, 20, 'sk', 'filled'); % add data points
    cb = colorbar; title(cb, '$\textrm{Q}_{in}$ [kW/m]','interpreter','latex','fontsize',14); %cb.Label.String = '$\alpha_{max}$';
    % clabel(contour_M, contour_c, [0.65:0.05:0.95], 'fontsize',14, 'interpreter','latex'); %,'Rotation',0 (can only use w/o contour_c)
    xlabel('$\mathrm{T_R}$~[${}^{\circ}$C]');
    ylabel('$\textrm{H}_{gap}$ [mm]');
    title(['$\textrm{V}_{in}$ = ', num2str(val_V_plot*1000,2), ' mm/s']);
    saveas(fig, ['contour_Qin_Hgap_Tr_Vin',replace(num2str(val_V_plot*1000),'.','p'),'.png']);
    
    % Curves for L_front vs (H_gap, T_R) for a given value of V_in
    fig = figure; box on; hold on;
    colororder(colors);
    scrn = get(0,'ScreenSize');
    set(gcf,'Position',[0.1*scrn(3), 0.1*scrn(4), 0.4*scrn(3), 0.42*scrn(4)]);
    % colororder()
    plts = [];
    for j = 1:size(H_gap,3)
        plt_i = plot(squeeze(T_R(:,ind_V_plot,j))-273.15, squeeze(L_front(:,ind_V_plot,j))./l_R(:,ind_V_plot,j), ['-',symbs{i}],'LineWidth',1.5,'DisplayName',['$\textrm{H}_{gap} = $',num2str(H_gap(1,1,j)*1000),'mm']);
        plts = [plts, plt_i];
    end
    xlabel('$\textrm{T}_R$~[${}^{\circ}$C]'); % xlabel('$\omega_R$ [rad/s]'); 
    ylabel('$\mathrm{L_{front}/\ell_R}$ ');
    title(['$\textrm{V}_{in}$ = ', num2str(val_V_plot*1000,2), ' mm/s']);
    legend(plts, 'location','eo');
    saveas(fig, ['plot_Lfront_Tr_Hgap_Vin',replace(num2str(val_V_plot*1000),'.','p'),'.png']);
    
end



