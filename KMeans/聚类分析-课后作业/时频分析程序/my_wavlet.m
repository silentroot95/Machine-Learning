function my_wavlet(filename,fs)
    close all;
    [t,s]=textread(filename,'%f,%f','headerlines',12); %�����ļ�ͷ���벨������  
    figure(2) %���Ʋ���ͼ
    plot(t*1000,s*1000,'color','red'); %ͼ��ʱ�䵥λת��Ϊ�����롱����ѹ��λת��Ϊ��������
    grid off;
    axis('tight')
    xlim([-2,11]*10^-4*1000);
    ylabel('��ѹ (mV)','fontsize',15)
    set(gca,'fontsize',15)
    xlabel('ʱ�� (ms)','fontsize',15)
    
    
    wavename='cmor3-3'; % ѡ��ʱƵ�����õ�С��
    totalscal=2^9;      % С���ֽ�߶�
    Fc=centfrq(wavename); % С��������Ƶ��
    c=2*Fc*totalscal;
    scals=c./(1:totalscal);
    f=scal2frq(scals,wavename,1/fs); % ���߶�ת��ΪƵ��
    coefs=cwt(s,scals,wavename); % ������С��ϵ��
    f=f/1000;
    h1=figure(1);
    set(h1,'position',[50,50,700,580])
    a2=axes('position',[0.12,0.1,0.13,0.6]);
    a3=axes('position',[0.32,0.1,0.67,0.6]);
    a1=axes('position',[0.32,0.78,0.67,0.2]);
    axes(a1);
    plot(t*1000,s*1000,'color','red');
    grid off;
    axis('tight')
    xlim([-2,11]*10^-4*1000);
    ylabel('��ѹ (mV)','fontsize',15)
    set(gca,'fontsize',15)
    axes(a3);
    load('MyColormaps','mycmap')
    contourf(t*1000,f,abs(coefs),4);
    set(h1,'Colormap',mycmap)
    set(gca,'YDir','normal')
    ylim([30,250])
    xlim([-2,11]*10^-4*1000);
    xlabel('ʱ�� (ms)','fontsize',15)
    set(gca,'fontsize',15)
    ch=colorbar;
    set(ch,'Location','East')
    axes(a2);
    [f,YY]=my_fft(s,fs);
    plot(YY/max(YY),f/1000,'linewidth',2);
    set(gca,'XDir','reverse')
    set(gca,'fontsize',15)
    ylabel('Ƶ�� (kHz)','fontsize',15)
    xlabel('����|Y|','fontsize',15)
    axis('tight')
    %xlim([0,15]*10^-3)
    ylim([30,250])
    
end
function [f,YY]=my_fft(y,Fs) %�ú������ź���fft�任�õ�Ƶ��ͼ
    L=length(y);
    NFFT = 2^nextpow2(L); % Next power of 2 from length of y
    Y = fft(y,NFFT)/L;
    f = Fs/2*linspace(0,1,NFFT/2);
    % Plot single-sided amplitude spectrum.
    YY=2*abs(Y(1:NFFT/2));
end  

