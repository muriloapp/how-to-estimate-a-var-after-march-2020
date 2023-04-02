clear all;
rng(10);
addpath([cd '/subroutines'])  
addpath([cd '/subroutines/DERIVESTsuite'])  
addpath([cd '/subroutines_additional'])  

%% load a monthly macro dataset 
[DATAMACRO,TEXTCPI] = xlsread('dataMay2021.xlsx','Monthly');
TimeMACRO = datetime(datestr((DATAMACRO(:,1)+datenum('12/31/1899','mm/dd/yy'))));
DataMACRO=DATAMACRO(:,2:end);

% some data transformations
DataMACRO(:,7)=exp(DataMACRO(:,7)/100);           % unemployment (because all variables are then logged and multiplied by 100)
DataMACRO(:,3)=DataMACRO(:,3)./DataMACRO(:,4);     % real pce, nominal PCE/PCE deflator
DataMACRO(:,6)=DataMACRO(:,6)./DataMACRO(:,1);    % real pce services, nominal PCE services/PCE services deflator

% variables in the baseline model
indmacro=[7 2 3 6 4 1 5]; 
series=["unemployment","employment","PCE","PCE: services","PCE (price)","PCE: services (price)","core PCE (price)"];
YLABELirf=["percentage points","100 x log points","100 x log points","100 x log points","100 x log points","100 x log points","100 x log points"];
YLABELfcst=["percentage points","index","index","index","index","index","index"];


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
Ylog = 100*log(Ylev);

Time = TimeMACRO(T0:end);
[T,n] = size(Ylog);

Tcovid=T-14;                      % first time period of COVID (March 2020)

%% standard monthly VAR to set hyperparameters
rng(10);            % random generator seed
lags=13;            % # VAR lags
ndraws=2*25000;      % # MCMC draws
res = bvarGLP_covid(Ylog,lags,'mcmc',1,'MCMCconst',1,'MNpsi',0,'sur',0,'noc',0,'Ndraws',ndraws,'hyperpriors',1,'Tcovid',Tcovid);


%% plot posterior of hyperparametersfigure('Position', [0, 0, 700, 600]);
figure('Position', [0, 0, 700, 600]);
subplot(3,2,1:2); histogram(res.mcmc.lambda,20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
title('\lambda  (MN prior)')
if ~isempty(Tcovid)
    subplot(3,2,3); histogram(res.mcmc.eta(:,1),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_0$','Interpreter','Latex')
    subplot(3,2,4); histogram(res.mcmc.eta(:,2),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_1$','Interpreter','Latex')
    subplot(3,2,5); histogram(res.mcmc.eta(:,3),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_2$','Interpreter','Latex')
    subplot(3,2,6); histogram(res.mcmc.eta(:,4),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\rho$','Interpreter','Latex')
end



%% experiment 1: IRFs to unemployment shock

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


%% conditional forecasts
Tjan2019=Tfeb2020-13;       % initial date for plots              
Tdec2022=Tfeb2020+46-12;    % end date for projections (end of 2022)
hmax=Tdec2022-T1av;         % corresponding maximum horizon of the forecast            

YYfcst=[100*log(DataMACRO(Tjan2019:T1av,indmacro));NaN(hmax,n)];  
YYfcst(end-hmax+1:end,1)=4.5+(5.8-4.5)*.955.^[0:hmax-1]';           % conditioning scenario from Blue Chip

if T1av==883;
    YYfcst(end-hmax+1:end,1)=[8.5;7.9;4.5+(7.9-4.5)*.955.^[0:hmax-3]']; % conditioning scenario if fcst performed in July 2020 
elseif T1av==882;
    YYfcst(end-hmax+1:end,1)=[10.2;8.5;7.9;4.5+(7.9-4.5)*.955.^[0:hmax-4]']; % conditioning scenario if fcst performed in June 2020
end


TTfcst=length(YYfcst);

DRAWSY=NaN(n,TTfcst,M);   % matrix to store draws of variables

% Forecasts
for i=1:M
    betadraw=squeeze(res.mcmc.beta(:,:,i));
    G=chol(squeeze(res.mcmc.sigma(:,:,i)))';

    if isempty(Tcovid);
        etapar=[1 1 1 1];
        tstar=1000000;
    else
        etapar=res.mcmc.eta(i,:); 
        tstar=TTfcst-hmax+Tcovid-T;
    end
    [varc,varZ,varG,varC,varT,varH]=FormCompanionMatrices(betadraw,G,etapar,tstar,n,lags,TTfcst);
    s00=flip(YYfcst(1:lags,:))'; s00=s00(:);
        
    P00=zeros(n*lags,n*lags);
    [DrawStates,shocks]=DisturbanceSmootherVAR(YYfcst,varc,varZ,varG,varC,varT,varH,s00,P00,TTfcst,n,n*lags,n,'simulation');
    DRAWSY(:,:,i)=DrawStates(1:n,:);
end

% plot of conditional forecasts
Dqqq=[.025 .16 .5 .84 .975];
IRFA=DRAWSY(1:n,:,:);
IRFAsorted=sort(IRFA,3);
    
figure('Position', [0, 0, 700, 600]);
count=0;
for ii=1:n
    count=count+1;
    if ii~=1;
        aux=exp(squeeze(IRFAsorted(ii,:,round(qqq*M)))/100); normalization=aux(13,3); aux=100*aux/normalization; % normalization must be changed
        realization=100*exp(log(DataMACRO(T1av+1:Tend,indmacro(ii))))/normalization;
    elseif ii==1;
        aux=squeeze(IRFAsorted(ii,:,round(qqq*M)));
    end
    subplot(ceil(n/2),2,count); quantilePlot([2019+.5/12:1/12:2022+11.5/12]', aux,ColorPlot); grid on; hold on    
    
    if ii~=1 & T1av==882;
        subplot(ceil(n/2),2,count); plot([2020+6.5/12:1/12:2020+8.5/12],realization,'k+','LineWidth',1.5)
    end
    
    ylabel(YLABELfcst(ii));
    title(series(ii));
    
    xlim([2019+.5/12 2022+11.5/12])
    xticks([2019+.5/12 2020+.5/12 2021+.5/12 2022+.5/12])
    xticklabels({'Jan 2019','Jan 2020','Jan 2021','Jan 2022'})
    
end

figure('Position', [0, 0, 700, 800]);
if ColorPlot==ColorCovid; count=1; elseif ColorPlot==ColorBase; count=2; end
for ii=1:n
    if ii~=1;
        aux=exp(squeeze(IRFAsorted(ii,:,round(qqq*M)))/100); normalization=aux(13,3); aux=100*aux/normalization; % normalization must be changed
        realization=100*exp(log(DataMACRO(T1av+1:Tend,indmacro(ii))))/normalization;
        
    elseif ii==1;
        aux=squeeze(IRFAsorted(ii,:,round(qqq*M)));
        realization=100*log(DataMACRO(T1av+1:Tend,indmacro(ii)));
    end
    subplot(7,2,count); quantilePlot([2019+.5/12:1/12:2022+11.5/12]', aux,ColorPlot); grid on; hold on    
    
    if ii~=1 & T1av==882;
        subplot(7,2,count); plot([2020+6.5/12:1/12:2020+8.5/12],realization,'k+','LineWidth',1.5)
    end
    
    xlim([2019+.5/12 2022+11.5/12])
    xticks([2019+.5/12 2020+.5/12 2021+.5/12 2022+.5/12])
    xticklabels({'Jan 2019','Jan 2020','Jan 2021','Jan 2022'})
    
    ylabel(YLABELfcst(ii));
    title(series(ii));
    count=count+2;
end
if ColorPlot==ColorCovid;
    subplot(7,2,1);
    text(2019,22,'Covid volatility - est. sample ends in 2020:6','FontSize',12,'FontWeight','bold','color',ColorPlot);
elseif ColorPlot==ColorBase;
    subplot(7,2,2);
    text(2018.85,22,'Constant volatility - est. sample ends in 2020:2','FontSize',12,'FontWeight','bold','color',ColorPlot);
end

if ColorPlot==ColorBase;
    for jj=1:2:13
        subplot(7,2,jj);
        yl=ylim;
        subplot(7,2,jj+1);
        ylim(yl);
    end
end


function [varc,varZ,varG,varC,varT,varH]=FormCompanionMatrices(betadraw,G,etapar,tstar,n,lags,TTfcst);
% forms the matrices of the VAR companion form

% matrices of observation equation
varc=zeros(n,TTfcst);
varZ=zeros(n,n*lags); varZ(:,1:n)=eye(n); varZ=repmat(varZ,1,1,TTfcst);
varG=repmat(zeros(n),1,1,TTfcst);

% matrices of state equation
B=betadraw;
varC=zeros(n*lags,1); varC(1:n)=B(1,:)';
varT=[B(2:end,:)';[eye(n*(lags-1)) zeros(n*(lags-1),n)]];
varH=zeros(n*lags,n,TTfcst); 
for t=1:TTfcst
    if t<tstar
        varH(1:n,1:end,t)=G;
    elseif t==tstar
        varH(1:n,1:end,t)=G*etapar(1);
    elseif t==tstar+1
        varH(1:n,1:end,t)=G*etapar(2);
    elseif t==tstar+2
        varH(1:n,1:end,t)=G*etapar(3);
    elseif t>tstar+2
        varH(1:n,1:end,t)=G*(1+(etapar(3)-1)*etapar(4)^(t-tstar-2));
    end
end
end