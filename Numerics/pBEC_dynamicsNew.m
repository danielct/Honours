clear all;
set(gcf,'Renderer','zbuffer')


%number of points
N=1024;
XYmax=55;
hspace=2*XYmax/(N-1);

dt=0.1; %step "time"


%Coefficients in the ODGPE: (normalised for  m_LP=10^{-3} me)
 g=0.187; %LP polariton-polariton scattering
 gR=2*g; %LP interaction with incoherent reservoir
 GammaC=1; GammaR=1.5*GammaC; % LP relaxation rate & %reservoir polaritons relaxation rate
 R=0.189; R=0.1; %stimulated scattering rate
 G=0.0; %direct interaction with the pump (initial blueshift)
 Pth = (GammaC * GammaR) / R;

%construct x,y numerical grid;

[x,y] = meshgrid(-XYmax:hspace:XYmax);
x_size=size(x);

%construct k-space numerical grid
%we are implicitly doing the fftshift
kvector=[0:N/2-1  -N/2:-1]*2*pi/(2*XYmax);
[kx, ky]=meshgrid(kvector, kvector);


%INITIAL SEED FOR GROUND STATE CONDENSATE:

% psi=0.2*(x.^2+y.^2).^(0).*exp(-0.04*(x.^2+y.^2));
% Start with a small random seed 
psi = randn(N, N);

%Pumping
%Pump=40*(exp(-0.003*((x+5).^2+(y).^2))); %Gaussian pump
% Pump=60*(exp(-0.01*((x+5).^2+(y).^2))); %Gaussian pump shifted horizontally
% Fairly narrow Gaussian pump whose peak power is a bit above threshold
Pump = 2.0 * Pth * exp(-0.005 * ((x).^2 + (y).^2));
%Initializing polariton wavefunction and reservoir density
psi0=psi;
n=0*Pump;

output=1;
%potential=-0.001*(exp(-0.01*((x).^2+(y).^2))).*(((x).^2+(y).^2)).^2; % toroidal trap

%potential=40*(exp(-0.01*((x).^2+(y).^2))); %Quadratic potential
potential = 0;
%potential=40*(exp(-0.01*((x).^2+(y).^2)))-40*(exp(-0.005*((x).^2+(y).^2)));

%Fourier transform of the kinetic energy term

K_vector=kx.^2+ky.^2;

kinetic_factor=exp(-1i*K_vector*dt);   
kinetic_factor_half=exp(-1i*K_vector*dt/2.0); 

%absorbing boundaries
damping=0.5*tanh(5.*(x+XYmax-5)).*tanh(5.*(y+XYmax-5))+0.5*tanh(5.*(-x+XYmax-5)).*tanh(5.*(-y+XYmax-5));

Intensity=[]; Time=[]; Energy=[];

fftw('planner','estimate');

for step = 1:600
    
    psi1=ifft2((kinetic_factor_half.*fft2(psi)));
    exp_factor_excitons=exp(-(GammaR+R*psi1.*conj(psi1))*dt);
    n1=n.*exp_factor_excitons+Pump*dt;
    exp_factor_polaritons=exp((-1i*potential-1i*g*psi1.*conj(psi1)-1i*gR*n1-1i*G*Pump+0.5*(R*n1-GammaC))*dt);
    psi_next=psi1.*exp_factor_polaritons;
    psi=ifft2((kinetic_factor_half.*fft2(psi_next))).*damping;
    n=n1;
   
    count=round(step/10);
    time=step*dt
    
    Intensity=[Intensity max(max(abs(psi).^2))];
    Time=[Time time];
    energy_stat=-sum(sum((1/4)*gradient(gradient(abs(psi).^2))-(1/2)*gradient(psi).*gradient(conj(psi))-(g*abs(psi).^2+gR*abs(n)+potential).*abs(psi).^2))./sum(sum(abs(psi).^2)); 
    Energy=[Energy energy_stat];
    
    
%     if count > output
%        output=count;
%        contourf(x,y,abs(psi).^2);
%      else dummy=1;
%      end
    
end  
plot(Time,Intensity); plot(Time,Energy);
mesh(x,y,abs(psi));  
contourf(x,y,abs(psi).^2)