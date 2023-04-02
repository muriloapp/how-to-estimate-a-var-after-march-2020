% setpriors.m 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file sets up the default choices for the priors of the BVAR of 
% Giannone, Lenza and Primiceri (2015), augmented with a change in 
% volatility at the time of Covid (March 2020).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONS:
%
% hyperpriors: 0 = no priors on hyperparameters
%              1 = reference priors on hyperparameters (default)
%
% Vc:       prior variance in the MN prior for the coefficients multiplying
%           the contant term (Default: Vc=10e6)
%           
% pos:      position of the variables that enter the VAR in first
%           differences and for which one might want to set the prior mean 
%           on the coefficient on the first own lag in the MN prior and the
%           prior mean of the sum-of-coefficients prior to 0 (instead of 1)
%           (Default: pos=[])
%
% Tcovid:   [] = the covid period is not included in the dataset (default)
%           position of observation corresponding to March 2020 or 2020:Q1
%
% MNalpha:  0 = Lag-decaying parameter of the MN prior set to 2 and
%               NOT treated as hyperparameter (default)
%           1 = Lag-decaying parameter of the MN prior treated as 
%               hyperparameter
%
% sur:      0 = single-unit-root prior is OFF
%           1 = single-unit-root prior is ON and its std is treated as an
%               hyperparameter (default)
%
% noc:      0 = no-cointegration (sum-of coefficients) prior is OFF
%           1 = no-cointegration (sum-of coefficients) is ON and its std is 
%               treated as an hyperparameter (default)
%
% fcast:    0 = does not generate forecasts at the posterior mode
%           1 = generates forecasts at the posterior mode (default)
%
% hz:       longest horizon at which the code generates forecasts
%           (default: maxhz=8)
%
% mcmc:     0 = does not run the MCMC (default)
%           1 = runs the MCMC after the maximization
%
% Ndraws:   number of draws in the MCMC (default: Ndraws=20000)
%
% Ndrawsdiscard: number of draws initially discarded to allow convergence 
%                in the in the MCMC (default=Ndraws/2)
%
% MCMCconst: scaling p for the MCMC (should be calibrated to achieve
%            an acceptance rate of approx 25%) (default: MCMCconst=1)
%
% MCMCfcast:   0 = does not generate forecasts when running the MCMC
%              1 = generates forecasts while running the MCMC
%                  (for each draw of the hyperparameters the code takes a 
%                  draw of the VAR coefficients and shocks, and generates 
%                  forecasts at horizons hz) (default).
%
% MCMCstorecoeff:   0 = does not store the MCMC draws of the VAR 
%                       coefficients and residual covariance matrix
%                   1 = stores the MCMC draws of the VAR coefficients and 
%                       residual covariance matrix (default)
%
% Last modified: 07/01/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Main options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if numarg>2
    for ii=1:2:size(varargin,2)
        eval([varargin{ii},'=', mat2str(varargin{ii+1})])
    end
end

%------------------------------
if ~exist('hyperpriors','var')
    hyperpriors=1;
end
r.setpriors.hyperpriors=hyperpriors;

%------------------------------
if ~exist('Vc','var')
    Vc=10e6;
end
r.setpriors.Vc=Vc;

%------------------------------
if ~exist('pos','var')
    pos=[];
end
r.setpriors.pos=pos;

%------------------------------
if ~exist('MNalpha','var')
    mn.alpha=0;
else
    mn.alpha=MNalpha;
end
r.setpriors.MNalpha=mn.alpha;

%------------------------------
if ~exist('Tcovid','var')
    Tcovid =[];
end;
r.Tcovid=Tcovid;

%------------------------------
if ~exist('sur','var')
    sur=1;
end
r.setpriors.sur=sur;

%------------------------------
if ~exist('noc','var')
    noc=1;
end
r.setpriors.noc=noc;

%------------------------------
if ~exist('Fcast','var')
    Fcast=1;
end
r.setpriors.Fcast=Fcast;

%------------------------------
if ~exist('hz','var')
    hz=[1:8]; 
else
    hz=[1:hz];
end
r.setpriors.hz=hz;

%------------------------------
if ~exist('mcmc','var')
    mcmc=0; 
end
r.setpriors.mcmc=mcmc;

%------------------------------
if ~exist('Ndraws','var')
    M=20000;
else
    M=Ndraws;
end
r.setpriors.Ndraws=M;

%------------------------------
if ~exist('Ndrawsdiscard','var')
    N=round(M/2);
else
    N=Ndrawsdiscard;
end
r.setpriors.Ndrawsdiscard=N;

%------------------------------
if ~exist('MCMCconst','var')
    const=1; 
else
    const=MCMCconst;
end
r.setpriors.MCMCconst=const;

%------------------------------
if ~exist('MCMCfcast','var')
    MCMCfcast=1; 
end
r.setpriors.MCMCfcast=MCMCfcast;

%------------------------------
if ~exist('MCMCstorecoeff','var')
    MCMCstorecoeff=1; 
end
r.setpriors.MCMCstorecoeff=MCMCstorecoeff;
%------------------------------

%% Other options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters of the hyperpriors, if choosen
if hyperpriors==1;
    
    mode.lambda=.2;     % hyperpriors modes
    mode.miu=1;
    mode.theta=1;
    
    sd.lambda=.4;       % hyperpriors std
    sd.miu=1;
    sd.theta=1;
    
    scalePSI=0.02^2;    % scale and shape of the IG on psi/(d-n-1)    
    
    priorcoef.lambda=GammaCoef(mode.lambda,sd.lambda,0);  % coefficients of hyperpriors
    priorcoef.miu=GammaCoef(mode.miu,sd.miu,0);
    priorcoef.theta=GammaCoef(mode.theta,sd.theta,0);
    
    %mode.eta=[10;100;.5];
    %sd.eta=[5;50;.25];
    %priorcoef.eta1=GammaCoef(mode.eta(1),sd.eta(1),0);
    %priorcoef.eta2=GammaCoef(mode.eta(2),sd.eta(2),0);
    
    mode.eta(4)=.8;
    sd.eta(4)=.2;
    mosd=[mode.eta(4) sd.eta(4)];
    fun=@(x) BetaCoef(x,mosd);
    albet=fsolve(fun,[2,2])';
    priorcoef.eta4.alpha=albet(1);    
    priorcoef.eta4.beta=albet(2);
  
else
    priorcoef=[];
end

% bounds for maximization
MIN.lambda = 0.0001;
MIN.alpha = 0.1;
MIN.theta = 0.0001;
MIN.miu = 0.0001;
MAX.lambda=5;
MAX.miu=50;
MAX.theta=50;
MAX.alpha=5;
MIN.eta(1:3)=[1;1;1];
MAX.eta(1:3)=[500;500;500];
MIN.eta(4)=.005;
MAX.eta(4)=.995;

function r=BetaCoef(x,mosd)
al=x(1);
bet=x(2);

mode=mosd(1);
sd=mosd(2);

r(1)=mode-(al-1)/(al+bet-2);
r(2)=sd-sqrt(al*bet/((al+bet)^2*(al+bet+1)));
end