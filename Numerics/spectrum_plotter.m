
% %the following gave so far the best result:
 %blueshift=0.5*gR*max(Pump)/GammaR;
 
 transform=fftshift(fft(M),1);
 omegaaxis=[-Tlength/2:(Tlength/2)-1]*2*pi/((Tlength)*dt);
 [kk, omega]=meshgrid(fftshift(kvector),omegaaxis);
 contourf(kk, omega,abs(transform).^2,200,'LineStyle','none'); 
 % ylim([0 10]); xlim([-4 4]);
 
 % print -depsc2 -adobecset -painter output.eps %the best vector output, small files & can remove layers in Illustrator
  

hold on;
% 
% %plot linear LP dispersion
plot(fftshift(kvector),fftshift(kvector).^2/2,'Color','b','LineWidth',10);
ylim([0 5]); xlim([-10 10]);


% %spectrum plotter in 2D
%contourf(fftshift(kx),fftshift(ky),abs(fftshift(fft2(psi))));
%ylim([-4 4]); xlim([-4 4]);