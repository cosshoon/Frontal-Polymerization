% Michael's custom MATLAB plot formatting
% Add to here if you discover new cool formatting that you think should be
% default in MATLAB
% To reset your workspace to the original MATLAB configuration, either
% close MATLAB, or use the command reset(groot)

function formatMike()

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(0,'DefaultAxesFontSize', 14);
set(groot,'DefaultTextarrowshapeInterpreter','latex');
set(groot,'DefaultTextarrowshapeFontsize',14);
set(groot,'DefaultTextboxshapeInterpreter','latex');
set(groot,'DefaultTextboxshapeFontsize',14);
set(groot,'DefaultTextFontSize',14);
set(groot,'DefaultColorbarTicklabelinterpreter','latex');
% set(0,'DefaultLineLineWidth',2);

% get(groot,'DefaultColorbarLabelInterpreter'); % <- apparently we can't
% set default properties to colorbar label :(
end