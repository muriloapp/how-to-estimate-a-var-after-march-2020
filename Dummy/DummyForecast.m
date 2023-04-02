% Computes full predictive density 
% using the BVAR of Giannone, Lenza and Primiceri (2012)

clear all;
rng(10);
addpath([cd '/subroutines'])  

%% load a monthly macro dataset 
[DATAMACRO,TEXTCPI] = xlsread('dataMay2021.xlsx','Monthly');
TimeMACRO = datetime(datestr((DATAMACRO(:,1)+datenum('12/31/1899','mm/dd/yy'))));
DataMACRO=DATAMACRO(:,2:end);

% some data transformations
DataMACRO(:,7)=exp(DataMACRO(:,7)/100);           % unemployment (because all variables are then logged and multiplied by 100)
DataMACRO(:,3)=DataMACRO(:,3)./DataMACRO(:,4);     % real pce, nominal PCE/PCE deflator
DataMACRO(:,6)=DataMACRO(:,6)./DataMACRO(:,1);    % real pce services, nominal PCE services/PCE services deflator

% variables in the baseline model 
indmacro=[7 2 3 6 4 1 5 8]; 
series=["unemployment","employment","PCE","PCE: services","PCE (price)","PCE: services (price)","core PCE (price)"];
YLABELirf=["percentage points","100 x log points","100 x log points","100 x log points","100 x log points","100 x log points","100 x log points"];
YLABELfcst=["percentage points","index","index","index","index","index","index"];

% Adding a dummy
TEXTCPI(1,8)={'Dummy'}
DataMACRO(1:410,8)=0;
DataMACRO(411:426,8)=1;

%% estimation sample (until May 2021)
Tend = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);       
Tfeb2020 = find(year(TimeMACRO)==2020 & month(TimeMACRO)==2);   % Feb 2020 observation (should not be modified)

T0 = find(year(TimeMACRO)==1989 & month(TimeMACRO)==1);         % beginning of estimation sample
T1estim = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);    % end of estimation sample
T1av = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);       % date of last available data for forecasts

ColorCovid=[.8941, .1020, .1098];
ColorBase=[44,127,184]./255;
ColorGrey=[.5 .5 .5];
ColorPlot=ColorCovid;

Ylev = DataMACRO(T0:T1estim,indmacro);
Ylog = [100*log(Ylev(:,1:7)) Ylev(:,8)];

Time = TimeMACRO(T0:end);
[T,n] = size(Ylog);

Tcovid=T-14; % first time period of COVID (March 2020)

lags = 13;

% Run the Bayesian VAR
res = bvarGLP(Ylog,lags,'mcmc',1,'sur',0,'noc',0,'MNalpha',0,'MNpsi',0,'MCMCconst',1);

figure(1)
hist(res.mcmc.lambda)
title('Posterior of the overall shrinckage of the MN prior')


%Computes the Impulse response function

%% IRFs to unemployment shock
% compute IRFs to an unemployment shock
H=60;
M = size(res.mcmc.beta,3);
Dirf1 = zeros(H+1,size(Ylog,2),M);
for jg = 1:M
    Dirf1(:,:,jg) =  bvarIrfs(res.mcmc.beta(:,:,jg),res.mcmc.sigma(:,:,jg),1,H+1);
end
sIRF1 = sort(Dirf1,3);
nn=7
% plot IRFs to an unemployment shock
qqq=[.025 .16 .5 .84 .975];     % percentiles of the posterior distribution
figure('Position', [0, 0, 700, 600]);
count=0;
for jn = 1:nn
    count=count+1;
    subplot(ceil(nn/2),2,count)
    quantilePlot([0:H]', squeeze(sIRF1(:,jn,round(qqq*M))),ColorPlot); hold on; grid on;
    line([0 H],[0 0],'color','k')
    xlabel('horizon');
    ylabel(YLABELirf(jn))
    title(series(jn));
end
