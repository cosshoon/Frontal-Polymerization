set(groot,'defaultFigureCreateFcn',@(fig,~)addToolbarExplorationButtons(fig));
set(groot,'defaultAxesCreateFcn',@(ax,~)set(ax.Toolbar,'Visible','off'));

addpath([userpath,'/formatPlots']);
addpath([userpath,'/pmkmp']);


clear all;