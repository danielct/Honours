%clear all;
set(gcf,'Renderer','zbuffer')


%number of points
N=1024;
XYmax=40;
hspace=2*XYmax/(N-1);
X=-2*XYmax*(1-1/N)/2:hspace:2*XYmax*(1-1/N)/2;  

dt=0.1; %step "time"

%nonlinearity: + for repulsive interaction
%Arbitrary parameters that work
%  g=0.002; %polariton self-interaction
%  gR=g; %interaction with reservoir
%  GammaC=0.1;
%  GammaR=1.5*GammaC;
%  G=0.0; %interaction with the pump
%  R=0.1;
 
%%These parameters give P_th~8, and most plots in the chiral paper are for these!
 g=0.187;
 gR=2*g;
 GammaC=1; GammaR=3.5*GammaC;
 R=0.187;
 G=0.0;
 
 
%  %Test parameters for weak exciton decay and inefficient R
%  %P_th=1
%  g=0;
%  gR=2;
%  GammaC=0.1; GammaR=0.05;
%  R=0.01;
%  G=0.0;
 
 

%Parameters of the Tokyo setup
 %g=6;
 %gR=2*g;
 %GammaC=0.02; GammaR=1.5*GammaC;
 %R=6.0;

%construct x,y space;

[x,y] = meshgrid(-XYmax:hspace:XYmax);
x_size=size(x);

%construct k-space

kvector=[0:N/2-1  -N/2:-1]*2*pi/(2*XYmax);
[kx, ky]=meshgrid(kvector, kvector);

%lattice parameters
%kl=2*XYmax/(120);
kl=0.5;
V0=0;
%lattice=0.0*V0*(sin(kl*x).^2+sin(kl*y).^2);
lattice=V0*(0.05*(x.^2+y.^2));
Phase=(x./sqrt(x.^2+y.^2)+1i*y./sqrt(x.^2+y.^2));
Phasem=(x./sqrt(x.^2+y.^2)-1i*y./sqrt(x.^2+y.^2));

%lattice=1+1+1*cos((3)*unwrap(angle(Phase))); %3-lobe combo


shift=0.0;

Phase0=((x-shift)./sqrt((x-shift).^2+y.^2)+1i*y./sqrt((x-shift).^2+y.^2));
Phase2=((x+shift)./sqrt((x+shift).^2+y.^2)+1i*y./sqrt((x+shift).^2+y.^2));

%single_charge=1.7*exp(-0.02*((x-shift).^2+y.^2)).*Phase0;
%single_charge_2=0.7*sqrt(((x+shift)/0.3).^2+(y/0.3).^2).*exp(-0.03*((x+shift).^2+(y).^2)).*Phase2;

%single_charge_2=1.7*exp(-0.02*((x+shift).^2+(y).^2)).*Phase2;
%load vortex_gap1_v30_mu345_offsite_14pi_256;

%INITIAL STATE FOR VORTEX GENERATION:

psip=0.002*(x.^2+y.^2).^(2).*exp(-0.05*(x.^2+y.^2)).*Phase;
psim=0.002*(x.^2+y.^2).^(2).*exp(-0.05*(x.^2+y.^2)).*Phasem;
%psi=psip;

%INITIAL STATE FOR GROUND STATE:

psi=0.002*exp(-0.0005*(x.^2+y.^2));

%INITIAL STATE FOR DIPOLE:

%psi=0.002*exp(-0.05*(x.^2+y.^2)).*sin(0.05*x);
%psi=psip+psim;



%Initial state for TILTED DIPOLE:
% psi_d=0.002*exp(-0.05*(x.^2+y.^2)).*sin(0.05*x);
% psi_d2=0.002*exp(-0.05*(x.^2+y.^2)).*sin(0.05*y);
% psi=psi_d+psi_d2;

%INITIAL STATE FOR vortex:

%psi=0.002*exp(-0.05*(x.^2+y.^2)).*Phase;



%psi=single_charge_2.*exp(i*2*x)+single_charge;
%psi=single_charge_2.*exp(i*x)+single_charge.*exp(-i*x);


%Pumping
%Pump=40*(exp(-0.001*((x).^2+(y).^2))).*(2+1*cos(3*unwrap(angle(Phase)))); %Gaussian pump with no OAM


%%%%%LG pumps interference%
%Pump=40*(exp(-0.001*((x).^2+(y).^2))).*(2+...
    %0.1*cos(1*unwrap(angle(Phase)))+...
    %0.2*cos(1*(unwrap(angle(Phase))+2*pi/3))+...
    %1*cos(3*unwrap(angle(Phase))));

    %%%%Three pump interference%%%%%%%
% Pump1=1.0*exp(-0.02*((x+10*cos(2*pi/3)).^2+(y+10*sin(-2*pi/3)).^2));
% Pump2=1.0*exp(-0.02*((x+10).^2+(y).^2));
% Pump3=1.0*exp(-0.02*((x+10*cos(2*pi/3)).^2+(y+10*sin(2*pi/3)).^2));
% Pump=100*(Pump1+Pump2+Pump3);


%%%%%Four pump interference

% Pump1=1.0*exp(-0.02*((x).^2+(y+10).^2));
% Pump2=1.0*exp(-0.02*((x+10).^2+(y).^2));
% Pump3=1.0*exp(-0.02*((x).^2+(y-10).^2));
% Pump4=1.0*exp(-0.02*((x-10).^2+(y).^2));
% Pump=100*(Pump1+Pump2+Pump3+Pump4);

%%%%%%Six pump interference "spiral" intensity
% Rpump=10; Wpump=0.02;
% Pump1=1.04*exp(-Wpump*((x+Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(-2*pi/6)).^2));
% Pump2=1.05*exp(-Wpump*((x+Rpump).^2+(y).^2)); 
% Pump3=1.0*exp(-Wpump*((x-Rpump).^2+(y).^2)); %to 0 for "laser beam"
% Pump4=1.03*exp(-Wpump*((x+Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(2*pi/6)).^2));               %to 0 for "laser beam"
% Pump5=1.01*exp(-Wpump*((x-Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(2*pi/6)).^2));
% Pump6=1.02*exp(-Wpump*((x-Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(-2*pi/6)).^2));
% Pump=22*(Pump1+Pump2+Pump3+Pump4+Pump5+Pump6);

% %%%%%Six pump interference equql intensity
% Rpump=20; Wpump=0.05;
% Pump1=1.00*exp(-Wpump*((x+Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(-2*pi/6)).^2));
% Pump2=1.00*exp(-Wpump*((x+Rpump).^2+(y).^2)); 
% Pump3=1.0*exp(-Wpump*((x-Rpump).^2+(y).^2));                                              %to 0 for "laser beam"
% Pump4=1.00*exp(-Wpump*((x+Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(2*pi/6)).^2));               %to 0 for "laser beam"
% Pump5=1.00*exp(-Wpump*((x-Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(2*pi/6)).^2));
% Pump6=1.00*exp(-Wpump*((x-Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(-2*pi/6)).^2));
% Pump=20*(Pump1+Pump2+Pump3+Pump4+Pump5+Pump6);

%%%%%%Six pump interference "accidently chiral" intensity
% Rpump=10; Wpump=0.02;
% Pump1=1.20*exp(-Wpump*((x+Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(-2*pi/6)).^2));
% Pump2=1.25*exp(-Wpump*((x+Rpump).^2+(y).^2)); 
% Pump3=1.00*exp(-Wpump*((x-Rpump).^2+(y).^2));                                              %to 0 for "laser beam"
% Pump4=1.15*exp(-Wpump*((x+Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(2*pi/6)).^2));               %to 0 for "laser beam"
% Pump5=1.05*exp(-Wpump*((x-Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(2*pi/6)).^2));
% Pump6=1.1*exp(-Wpump*((x-Rpump*cos(2*pi/6)).^2+(y+Rpump*sin(-2*pi/6)).^2));
% Pump=22*(Pump1+Pump2+Pump3+Pump4+Pump5+Pump6);

%%%%%%Six pump interference "spiral" position
Rpump=9.5; Wpump=0.02;
 b=0.135; %increasing this increases pitch/asymmetry of the spiral
 numSpots = 6;
Pump = cell(numSpots,1);
for n = 1:numSpots
    theta=2*pi/6*n;
 xshift=(Rpump+b*theta)*cos(theta);
 yshift=(Rpump+b*theta)*sin(theta);
    %Pump{n} = 1.0*exp(-Wpump*((x+xshift).^2+(y+yshift).^2)).*exp(i*kv*sqrt((y+yshift).^2+(x+xshift).^2));
    Pump{n} = 1.0*exp(-Wpump*((x+xshift).^2+(y+yshift).^2));
    %%non-interfering pump spots for testing
end
 Pump=23*(Pump{1}+Pump{2}+Pump{3}+Pump{4}+Pump{5}+Pump{6});






%Pump=100*(2.0*exp(-0.02*((x+20).^2+(y).^2))+1.0*exp(-0.02*((x-20).^2+(y).^2)));


%Pump=20*((0.005*x.^2+0.005*y.^2)).^(4).*exp(-0.005*(x.^2+y.^2));

%Pump=100*((0.005*x.^2+0.01*y.^2)).^(2).*exp(-(0.005*x.^2+0.01*y.^2));


%Pump=120*ones(size(x.^2+y.^2));
%Pump=0.2*(sin(kl*x).^2+sin(kl*y).^2);

psi0=psi;
n=0*Pump;

%psi=psi.*exp(i*0.5*x+i*0*y);
output=0;

K_vector=kx.^2+ky.^2;

kinetic_factor=exp(-1i*K_vector*dt);   
kinetic_factor_half=exp(-1i*K_vector*dt/2.0); 

%absorbing boundaries

damping=0.5*tanh(1.*(x+35)).*tanh(1.*(y+35))+0.5*tanh(1.*(-x+35)).*tanh(1.*(-y+35));

Amplitude=[]; Time=[]; Mu=[]; 

% Mustat=[];A
       
fftw('planner','estimate');
%Integrating for initial transient period
%Tinitial=1000;
Tinitial=100;
for step = 1:Tinitial
    
    psiprev=psi;
    
    psi1=ifft2((kinetic_factor_half.*fft2(psi)));
    exp_factor_excitons=GammaR+R*psi1.*conj(psi1);
    
    %EXACT STEP FOR nR
    n1=n.*exp(-exp_factor_excitons*dt)+(Pump./exp_factor_excitons).*(1-exp(-exp_factor_excitons*dt));
     

    %RK4 STEP FOR nR
%     k1=dt*(Pump-exp_factor_excitons.*n);
%     k2=dt*(Pump-exp_factor_excitons.*(n+k1/2));
%     k3=dt*(Pump-exp_factor_excitons.*(n+k2/2));
%     k4=dt*(Pump-exp_factor_excitons.*(n+k3));
%     n1=n+(1/6)*(k1+2*k2+2*k3+k4);
    
    exp_factor_polaritons=exp((-1i*lattice-1i*g*psi1.*conj(psi1)-1i*gR*n1-1i*G*Pump+0.5*(R*n1-GammaC))*dt);
    psi_next=psi1.*exp_factor_polaritons;
    psi=ifft2((kinetic_factor_half.*fft2(psi_next))).*damping;
    n=n1;
   
    count=round(step/50);
    time=step*dt
    Ampnext=max(max(abs(psi)));
    
    %mu=-1i*(log(sum(sum(psiprev))/sum(sum(psi))))/(2*dt);
    
    mustat=-sum(sum((1/4)*gradient(gradient(abs(psi).^2))-(1/2)*gradient(psi).*gradient(conj(psi))-(g*abs(psi).^2+gR*abs(n)+lattice).*abs(psi).^2))./sum(sum(abs(psi).^2)); 
    
    Amplitude=[Amplitude Ampnext^2];
    Time=[Time time];
    Mu=[Mu mustat];
    
    
    %Mustat=[Mustat mustat];
    %if count > output
     %   output=count;
     %   mesh(x,y,real(psi));
     %else dummy=1;
     %end
   
%  %Movie output:   
   hf=figure(1);
%   set(gcf,'Color','w', 'Position',[300 200 600 300]);
   %subplot('Position',[0 0 1/2 1])
   imagesc(X,-X,abs(psi))
  axis xy
   axis(5*[-5 5 -5 5])
   axis square
   axis off
%   %text(-1,-4,['Time: ' num2str(t,'%5.2f')],'FontName','TimesNewRoman','FontSize',12,'Color','w')
%   %caxis(max(max(abs(Pump)))*[0 1])
%   hold on
%   %line(2*[-1 1]*b*cos(om*t), 2*[-1 1]*b*sin(om*t),'Color','w','LineWidth',1);
%  subplot('Position',[1/2 0 1/2 1])
%   imagesc(X,-X,angle(psi))
%   %caxis(pi*[-1 1])
%     axis xy
%   %axis(5*[-1 1 -1 1])
%   axis square
%   axis off
     
     
end  

%second time loop for collecting spectral data into matrix M. longer time =
%narrower spectrum

%stepstart=step;
%Tfinal=10500;
%Tlength=Tfinal-Tinitial;
Tlength = 500;
Tfinal = Tinitial + Tlength;
M=[]; 

for step = Tinitial+1:Tfinal
    
    psiprev=psi;
    
    Ktransform=fft2(psi);
    M=[M; fftshift(Ktransform(:,512))'];
    
    psi1=ifft2((kinetic_factor_half.*Ktransform));
    exp_factor_excitons=GammaR+R*psi1.*conj(psi1);
    
    %EXACT STEP FOR nR
    n1=n.*exp(-exp_factor_excitons*dt)+(Pump./exp_factor_excitons).*(1-exp(-exp_factor_excitons*dt));
     

    %RK4 STEP FOR nR
%     k1=dt*(Pump-exp_factor_excitons.*n);
%     k2=dt*(Pump-exp_factor_excitons.*(n+k1/2));
%     k3=dt*(Pump-exp_factor_excitons.*(n+k2/2));
%     k4=dt*(Pump-exp_factor_excitons.*(n+k3));
%     n1=n+(1/6)*(k1+2*k2+2*k3+k4);
    
    exp_factor_polaritons=exp((-1i*lattice-1i*g*psi1.*conj(psi1)-1i*gR*n1-1i*G*Pump+0.5*(R*n1-GammaC))*dt);
    psi_next=psi1.*exp_factor_polaritons;
    psi=ifft2((kinetic_factor_half.*fft2(psi_next))).*damping;
    n=n1;
   
    count=round(step/50);
    time=step*dt
    Ampnext=max(max(abs(psi)));
    
    %mu=-1i*(log(sum(sum(psiprev))/sum(sum(psi))))/(2*dt);
    
    mustat=-sum(sum((1/4)*gradient(gradient(abs(psi).^2))-(1/2)*gradient(psi).*gradient(conj(psi))-(g*abs(psi).^2+gR*abs(n)+lattice).*abs(psi).^2))./sum(sum(abs(psi).^2)); 
    
    Amplitude=[Amplitude Ampnext^2];
    Time=[Time time];
    Mu=[Mu mustat];
    
    
    %Mustat=[Mustat mustat];
    %if count > output
     %   output=count;
     %   mesh(x,y,real(psi));
     %else dummy=1;
     %end
   
%  %Movie output:   
   hf=figure(1);
%   set(gcf,'Color','w', 'Position',[300 200 600 300]);
   %subplot('Position',[0 0 1/2 1])
   imagesc(X,-X,abs(psi))
  axis xy
   axis(5*[-5 5 -5 5])
   axis square
   axis off
%   %text(-1,-4,['Time: ' num2str(t,'%5.2f')],'FontName','TimesNewRoman','FontSize',12,'Color','w')
%   %caxis(max(max(abs(Pump)))*[0 1])
%   hold on
%   %line(2*[-1 1]*b*cos(om*t), 2*[-1 1]*b*sin(om*t),'Color','w','LineWidth',1);
%  subplot('Position',[1/2 0 1/2 1])
%   imagesc(X,-X,angle(psi))
%   %caxis(pi*[-1 1])
%     axis xy
%   %axis(5*[-1 1 -1 1])
%   axis square
%   axis off
     
     
end  


plot(Time,Mu);
%print('-deps2','amp_vs_time_P6_V0_t70'); 
%mesh(x,y,abs(psi));  
figure;
contourf(x,y,abs(psi).^2,'LineStyle', 'none'); colorbar('FontSize', 10); axis square; 
xlim([-25 25]); ylim([-25 25]); 
              
Integral=sum(sum((R*n-GammaC).*abs(psi).^2))*hspace^2;

mustat=-sum(sum((1/4)*gradient(gradient(abs(psi).^2))-(1/2)*gradient(psi).*gradient(conj(psi))-(g*abs(psi).^2+gR*abs(n)+lattice).*abs(psi).^2))./sum(sum(abs(psi).^2));
%print('-deps2','density_P6_V0_t70'); 