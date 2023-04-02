function [sdraw,epsdraw]=DisturbanceSmootherVAR(y,c,Z,G,C,B,H,s00,P00,T,n,ns,ne,SS);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs a draws from the posterior of the disturbances
% and unobservable states of the following state-space model
%
% y(t) = c + Z * s(t) + G * me(t)  ~ N(0,I)
% s(t) = C + B s(t-1) + H * eta(t) ~ N(0,I)
% s(0) ~ N(s00,P00)
% y(t)   is nx1 
% s(t)   is nsx1
% eta(t) is nex1
% SS     must be set to either "simulation" or "smoother"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(Z,3)==1; Z=repmat(Z,1,1,T); end
if size(c,2)==1; c=repmat(c,1,T); end
if size(G,3)==1; G=repmat(G,1,1,T); end
if size(H,3)==1; H=repmat(H,1,1,T); end
ind=isfinite(y);

V=zeros(n,T);
K=zeros(ns,n,T);
HINV=zeros(n,n,T);
SHAT=zeros(ns,T);
SIG=zeros(ns,ns,T);
shat=s00;
sig=P00;

for t=1:T
    [shat,sig,v,k,hinv]=kfilter_forDS_VAR(y(t,ind(t,:))',c(ind(t,:),t),squeeze(Z(ind(t,:),:,t)),...
        squeeze(G(ind(t,:),:,t)),C,B,squeeze(H(:,:,t)),shat,sig);
    SHAT(:,t)=shat; SIG(:,:,t)=sig;
    V(ind(t,:),t)=v; K(:,ind(t,:),t)=k; HINV(ind(t,:),ind(t,:),t)=hinv;
end

% disturbance smoother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epshat=zeros(ne,T);
r=zeros(ns,1);
for t=T:-1:1
    epshat(:,t)=squeeze(H(:,:,t))'*squeeze(Z(ind(t,:),:,t))'*squeeze(HINV(ind(t,:),ind(t,:),t))*V(ind(t,:),t)+...
        squeeze(H(:,:,t))'*(eye(ns)-squeeze(K(:,ind(t,:),t))*squeeze(Z(ind(t,:),:,t)))'*r;
    r=B'*squeeze(Z(ind(t,:),:,t))'*squeeze(HINV(ind(t,:),ind(t,:),t))*V(ind(t,:),t)+...
        B'*(eye(ns)-squeeze(K(:,ind(t,:),t))*squeeze(Z(ind(t,:),:,t)))'*r;
end

if strcmp(SS,'smoother')
    % smoothed states
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sdraw=zeros(ns,T);
    sdraw(:,1)=C + B*s00 + squeeze(H(:,:,1))*epshat(:,1);
    for t=2:T
        sdraw(:,t)=C + B*sdraw(:,t-1) + squeeze(H(:,:,t))*epshat(:,t);
    end
    epsdraw=[];
    
elseif strcmp(SS,'simulation')
    % simulating new shocks, states and observables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    epsplus=randn(ne,T);
    splus=zeros(ns,T);
    yplus=zeros(n,T);
    splus(:,1)=C + B*s00 + squeeze(H(:,:,1))*epsplus(:,1);
    yplus(ind(t,:),1)=squeeze(Z(ind(1,:),:,1))*splus(:,1);
    for t=2:T
        splus(:,t)=C + B*splus(:,t-1) + squeeze(H(:,:,t))*epsplus(:,t);
        yplus(ind(t,:),t)=squeeze(Z(ind(t,:),:,t))*splus(:,t);
    end
    
    % Kalman filter
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Vplus=zeros(n,T);
    Kplus=zeros(ns,n,T);
    HINVplus=zeros(n,n,T);
    SHATplus=zeros(ns,T);
    SIGplus=zeros(ns,ns,T);
    shat=s00;
    sig=P00;
    
    for t=1:T
        [shat,sig,v,k,hinv]=kfilter_forDS_VAR(yplus(ind(t,:),t),c(ind(t,:),t),squeeze(Z(ind(t,:),:,t)),...
            squeeze(G(ind(t,:),:,t)),C,B,squeeze(H(:,:,t)),shat,sig);
        SHATplus(:,t)=shat; SIGplus(:,:,t)=sig;
        Vplus(ind(t,:),t)=v; Kplus(:,ind(t,:),t)=k; HINVplus(ind(t,:),ind(t,:),t)=hinv;
    end
    
    % disturbance smoother
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    epshatplus=zeros(ne,T);
    r=zeros(ns,1);
    for t=T:-1:1
        %epshatplus(:,t)=H'*Z'*squeeze(HINVplus(:,:,t))*Vplus(:,t)+H'*(eye(ns)-squeeze(Kplus(:,:,t))*Z)'*r;
        %r=B'*Z'*squeeze(HINVplus(:,:,t))*Vplus(:,t)+B'*(eye(ns)-squeeze(Kplus(:,:,t))*Z)'*r;
        
        epshatplus(:,t)=squeeze(H(:,:,t))'*squeeze(Z(ind(t,:),:,t))'*squeeze(HINVplus(ind(t,:),ind(t,:),t))*Vplus(ind(t,:),t)+...
            squeeze(H(:,:,t))'*(eye(ns)-squeeze(Kplus(:,ind(t,:),t))*squeeze(Z(ind(t,:),:,t)))'*r;
        r=B'*squeeze(Z(ind(t,:),:,t))'*squeeze(HINVplus(ind(t,:),ind(t,:),t))*Vplus(ind(t,:),t)+...
            B'*(eye(ns)-squeeze(Kplus(:,ind(t,:),t))*squeeze(Z(ind(t,:),:,t)))'*r;
        
    end
    
    epsdraw=epshat+epsplus-epshatplus;
    
    sdraw=zeros(ns,T);
    sdraw(:,1)=C + B*s00 + squeeze(H(:,:,1))*epsdraw(:,1);
    for t=2:T
        sdraw(:,t)=C + B*sdraw(:,t-1) + squeeze(H(:,:,t))*epsdraw(:,t);
    end
end
