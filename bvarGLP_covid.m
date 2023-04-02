function r = bvarGLP_covid(y,lags,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function estimates the BVAR of Giannone, Lenza and Primiceri (2015),
% augmented with a change in volatility at the time of Covid (March 2020).
% The current version is designed to be run on monthly data. The path of
% common volatility is controlled by 3 hyperparameters, and has the form
% [eta(1) eta(2)*eta(3)^[0:end]]
%
% y:        data matrix
% lags:     number of lags in the VAR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% set BVAR priors (several options available, see setpriors.m)
%  if varargin=[] --> default settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numarg = nargin;
setpriors_covid;


%% data matrix manipulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dimensions
[TT,n]=size(y);
k=n*lags+1;         % # coefficients for each equation


% constructing the matrix of regressors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=zeros(TT,k);
x(:,1)=1;
for i=1:lags
    x(:,1+(i-1)*n+1:1+i*n)=lag(y,i);
end

y0=mean(y(1:lags,:),1);
x=x(lags+1:end,:);
y=y(lags+1:end,:);
[T,n]=size(y);
Tcovid=Tcovid-lags;


% MN prior mean
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b=zeros(k,n);
diagb=ones(n,1);
diagb(pos)=0;   % Set to zero the prior mean on the first own lag for variables selected in the vector pos
b(2:n+1,:)=diag(diagb);


%% starting values for the minimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda0=.2;     % std of MN prior
theta0=1;       % std of sur prior
miu0=1;         % std of noc prior
alpha0=2;       % lag-decaying parameter of the MN prior
aux=mean(abs(y(Tcovid:max([Tcovid+1 T]),:)-y(Tcovid-1:max([Tcovid+1 T])-1,:))',1)./...
    mean(mean(abs(y(2:Tcovid-1,:)-y(1:Tcovid-2,:))));
if isempty(aux)
    eta0=[];
elseif length(aux)==2;
    eta0=[aux';aux(1);.8];      % volatility hyperparameters 
elseif length(aux)>=3;
    eta0=[aux(1:3)';.8];             % volatility hyperparameters 
end


% residual variance of AR(1) for each variable
SS=zeros(n,1);
for i=1:n
    Tend=T; if ~isempty(Tcovid); Tend=Tcovid-1; end
    ar1=ols1(y(2:Tend,i),[ones(Tend-1,1),y(1:Tend-1,i)]);
    SS(i)=ar1.sig2hatols;
end

inlambda=-log((MAX.lambda-lambda0)/(lambda0-MIN.lambda));
inHlambda=(1/(MAX.lambda-lambda0)+1/(lambda0-MIN.lambda))^2*(abs(lambda0)/1)^2;

if mn.alpha==1;
    inalpha=-log((MAX.alpha-alpha0)/(alpha0-MIN.alpha));
    inHalpha=(1/(MAX.alpha-alpha0)+1/(alpha0-MIN.alpha))^2*(abs(alpha0)/1)^2;
elseif mn.alpha==0;
    inalpha=[];
    inHalpha=[];
end

if sur==1;
    intheta=-log((MAX.theta-theta0)/(theta0-MIN.theta));
    inHtheta=(1/(MAX.theta-theta0)+1/(theta0-MIN.theta))^2*(abs(theta0)/1)^2;
elseif sur==0;
    intheta=[];
    inHtheta=[];
end

if noc==1;
    inmiu=-log((MAX.miu-miu0)/(miu0-MIN.miu));
    inHmiu=(1/(MAX.miu-miu0)+1/(miu0-MIN.miu))^2*(abs(miu0)/1)^2;
elseif noc==0;
    inmiu=[];
    inHmiu=[];
end

if ~isempty(Tcovid)
    ncp=length(eta0);           % # "covid" hyperparameters
    ineta=-log((MAX.eta'-eta0)./(eta0-MIN.eta'));
    inHeta=(1./(MAX.eta'-eta0)+1./(eta0-MIN.eta')).^2.*(abs(eta0)/1).^2;
else
    ineta=[];
    inHeta=[];
end

x0=[inlambda;ineta;intheta;inmiu;inalpha];
H0=diag([inHlambda;inHeta;inHtheta;inHmiu;inHalpha]); % initial guess for the inverse Hessian

%% maximization of the posterior of the hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[fh,xh,gh,H,itct,fcount,retcodeh] = csminwel('logMLVAR_formin_covid',x0,H0,[],1e-16,1000,y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,hyperpriors,priorcoef,Tcovid);


%% output of the maximization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VAR coefficients and residuals at the posterior model
[fh,r.postmax.betahat,r.postmax.sigmahat]=logMLVAR_formin_covid(xh,y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,hyperpriors,priorcoef,Tcovid);

r.lags = lags;                      % # lags
r.postmax.itct=itct;                % #iteration before reaching maximum
r.postmax.SSar1=SS;                 % residual variance of AR(1) for each variable
r.postmax.logPost=-fh;              % value of the posterior of the hyperparameters at the peak
r.postmax.lambda=MIN.lambda+(MAX.lambda-MIN.lambda)/(1+exp(-xh(1)));    % std of MN prior at the peak

r.postmax.theta=MAX.theta;
r.postmax.miu=MAX.miu;
r.postmax.eta=MAX.eta';


if ~isempty(Tcovid)
    % covid-volatility hyperparameters
    r.postmax.eta=MIN.eta'+(MAX.eta'-MIN.eta')./(1+exp(-xh(2:ncp+1)));   
    
    if sur==1;
        % std of sur prior at the peak
        r.postmax.theta=MIN.theta+(MAX.theta-MIN.theta)/(1+exp(-xh(ncp+2)));
        if noc==1;
            % std of noc prior at the peak
            r.postmax.miu=MIN.miu+(MAX.miu-MIN.miu)/(1+exp(-xh(ncp+3)));
        end
    elseif sur==0;
        if noc==1;
            % std of sur prior at the peak
            r.postmax.miu=MIN.miu+(MAX.miu-MIN.miu)/(1+exp(-xh(ncp+2)));
        end
    end
else
    r.postmax.eta(1:3)=1;
    if sur==1;
        % std of sur prior at the peak
        r.postmax.theta=MIN.theta+(MAX.theta-MIN.theta)/(1+exp(-xh(2)));
        if noc==1;
            % std of sur prior at the peak
            r.postmax.miu=MIN.miu+(MAX.miu-MIN.miu)/(1+exp(-xh(3)));
        end
    elseif sur==0;
        if noc==1;
            % std of sur prior at the peak
            r.postmax.miu=MIN.miu+(MAX.miu-MIN.miu)/(1+exp(-xh(2)));
        end
    end
end

if mn.alpha==0;
    r.postmax.alpha=2;
elseif mn.alpha==1;
    % Lag-decaying parameter of the MN prior
    r.postmax.alpha=MIN.alpha+(MAX.alpha-MIN.alpha)/(1+exp(-xh(end)));
end


%% forecasts at the posterior mode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Fcast==1
    Y=[y;zeros(hz(end),n)];
    for tau=1:max(hz)
        xT=[1;reshape(Y([T+tau-1:-1:T+tau-lags],:)',k-1,1)]';
        Y(T+tau,:)=xT*r.postmax.betahat;
    end
    r.postmax.forecast=Y(T+hz,:);
end


%% MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mcmc==1
    
    % old computation of the inverse Hessian
%     % Jacobian of the transformation of the hyperparameters that has been
%     % used for the constrained maximization
%     JJ=exp(xh)./(1+exp(xh)).^2;
%     JJ(1)=(MAX.lambda-MIN.lambda)*JJ(1);
%     
%     if ~isempty(Tcovid)
%         
%         JJ(2:ncp+1)=(MAX.eta'-MIN.eta').*JJ(2:ncp+1);
%         
% %         JJ(2:3)=exp(xh(2:3))
% %         JJ(1+ncp)=(MAX.eta(3)-MIN.eta(3)).*JJ(1+ncp);
%         
%         if sur==1;
%             JJ(ncp+2)=(MAX.theta-MIN.theta)*JJ(ncp+2);
%             if noc==1;
%                 JJ(ncp+3)=(MAX.miu-MIN.miu)*JJ(ncp+3);
%             end
%         elseif sur==0;
%             if noc==1;
%                 JJ(ncp+2)=(MAX.miu-MIN.miu)*JJ(ncp+2);
%             end
%         end
%     else
%         if sur==1;
%             JJ(2)=(MAX.theta-MIN.theta)*JJ(2);
%             if noc==1;
%                 JJ(3)=(MAX.miu-MIN.miu)*JJ(3);
%             end
%         elseif sur==0;
%             if noc==1;
%                 JJ(2)=(MAX.miu-MIN.miu)*JJ(2);
%             end
%         end
%     end
%     
%     if mn.alpha==1;
%         JJ(end)=(MAX.alpha-MIN.alpha)*JJ(end);
%     end
%     
%     JJ=diag(JJ);
%     HH=JJ*H*JJ';
%     
%     % regularizing the Hessian (making sure it is positive definite)
%     [V,E]=eig(HH);
%     HH=V*abs(E)*V';
    
    % recovering the posterior mode
    if ~isempty(Tcovid)
        modeeta=r.postmax.eta;
    else
        modeeta=[];
    end
    
    if mn.alpha==1;
        modealpha=r.postmax.alpha;
    elseif mn.alpha==0;
        modealpha=[];
    end
    
    if sur==1;
        modetheta=r.postmax.theta;
    elseif sur==0;
        modetheta=[];
    end
    
    if noc==1;
        modemiu=r.postmax.miu;
    elseif noc==0;
        modemiu=[];
    end
    
    postmode=[r.postmax.lambda;modeeta;modetheta;modemiu;modealpha];
    
    % new computation of the inverse Hessian
    fun = @(par) logMLVAR_formcmc_covid(par,y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,0,hyperpriors,priorcoef,Tcovid);
    Hess = hessian(fun,postmode);
    [V,E]=eig(Hess);
    HH=-inv(Hess);
    if ~isempty(Tcovid) & T<=Tcovid+1
        HessNew=Hess;
        HessNew(4,:)=0; HessNew(:,4)=0; HessNew(4,4)=-1; 
        HH=-inv(HessNew); HH(4,4)=HH(2,2);
    end
    
    
    % starting value of the Metropolis algorithm
    P=zeros(M,length(xh));
    logMLold=-10e15;
    while logMLold==-10e15
        P(1,:)=mvnrnd(postmode,HH*const^2,1);
        [logMLold,betadrawold,sigmadrawold]=logMLVAR_formcmc_covid(P(1,:)',y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,...
            max([MCMCfcast,MCMCstorecoeff]),hyperpriors,priorcoef,Tcovid);
    end
    
    % matrix to store the draws of the VAR coefficients if MCMCstorecoeff is on
    if MCMCstorecoeff==1
        r.mcmc.beta  = zeros(k,n,M-N);
        r.mcmc.sigma = zeros(n,n,M-N);
    end
    
    % matrix to store the forecasts if MCMCfcast is on
    if MCMCfcast==1
        r.mcmc.Dforecast=zeros(length(hz),n,M-N);
    end
    
    % Metropolis iterations
    count=0;
    for i=2:M
        if i==100*floor(.01*i);
                        disp(['Now running the ',num2str(i),'th mcmc iteration (out of ',num2str(M),')'])
        end
        % draw candidate value
        P(i,:)=mvnrnd(P(i-1,:),HH*const^2,1);
        [logMLnew,betadrawnew,sigmadrawnew]=logMLVAR_formcmc_covid(P(i,:)',y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,...
            max([MCMCfcast,MCMCstorecoeff]),hyperpriors,priorcoef,Tcovid);
        
        if logMLnew>logMLold
            logMLold=logMLnew;
            count=count+1;
        else
            if rand(1)<exp(logMLnew-logMLold);
                logMLold=logMLnew;
                count=count+1;
            else
                P(i,:)=P(i-1,:);
                % if MCMCfcast is on, take a new draw of the VAR coefficients
                % with the old hyperparameters if have rejected the new ones
                % (the speed of this step could be probably improved)
                if MCMCfcast==1 | MCMCstorecoeff==1
                    [junk,betadrawnew,sigmadrawnew]=logMLVAR_formcmc_covid(P(i,:)',y,x,lags,T,n,b,MIN,MAX,SS,Vc,pos,mn,sur,noc,y0,...
                        max([MCMCfcast,MCMCstorecoeff]),hyperpriors,priorcoef,Tcovid);
                end
            end
        end
        
        % stores draws of VAR coefficients if MCMCstorecoeff is on
        if i>N & MCMCstorecoeff==1     
            r.mcmc.beta(:,:,i-N) = betadrawnew;  
            r.mcmc.sigma(:,:,i-N) = sigmadrawnew;   
        end
        
        % produce and store the forecasts if MCMCfcast is on
        if i>N & MCMCfcast==1
            Y=[y;zeros(hz(end),n)];
            for tau=1:max(hz)
                xT=[1;reshape(Y([T+tau-1:-1:T+tau-lags],:)',k-1,1)]';
                if ~isempty(Tcovid)
                    if T==Tcovid
                        scaling=P(i,3)*(tau==1)+(1+(P(i,4)-1)*P(i,5)^(tau-2))*(tau>=2);
                    elseif T>Tcovid
                        scaling=(1+(P(i,4)-1)*P(i,5)^(T-Tcovid+tau-2));
                    end                  
                else
                    scaling=1;;
                end
                errors=mvnrnd(zeros(1,n),sigmadrawnew)*scaling;
                Y(T+tau,:)=xT*betadrawnew+errors;  
            end
            r.mcmc.Dforecast(:,:,i-N)=Y(T+hz,:);
            %r.IRF(:,:,:,i) = IRF(betadrawnew,sigmadrawnew);
        end
        
    end
    
    % store the draws of the hyperparameters
    r.mcmc.lambda=P(N+1:end,1);   % std MN prior
    
    if ~isempty(Tcovid)
        % diagonal elements of the scale matrix of the IW prior on the residual variance
        r.mcmc.eta=P(N+1:end,[2:ncp+1]);
        if sur==1;
            % std of sur prior
            r.mcmc.theta=P(N+1:end,ncp+2);
            if noc==1;
                % std of noc prior
                r.mcmc.miu=P(N+1:end,ncp+3);
            end
        elseif sur==0;
            if noc==1;
                % std of noc prior
                r.mcmc.miu=P(N+1:end,ncp+2);
            end
        end
    else
        if sur==1;
            % std of sur prior
            r.mcmc.theta=P(N+1:end,2);
            if noc==1;
                % std of noc prior
                r.mcmc.miu=P(N+1:end,3);
            end
        elseif sur==0;
            if noc==1;
                % std of noc prior
                r.mcmc.miu=P(N+1:end,2);
            end
        end
    end
    
    if mn.alpha==1;
        % Lag-decaying parameter of the MN prior
        r.mcmc.alpha=P(N+1:end,end);
    end
    
    % acceptance rate
    r.mcmc.ACCrate=mean(r.mcmc.lambda(2:end)~=r.mcmc.lambda(1:end-1));
     
end