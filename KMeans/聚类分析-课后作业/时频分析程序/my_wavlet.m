function my_wavlet(filename,fs)
    close all;
    [t,s]=textread(filename,'%f,%f','headerlines',12); %跳过文件头读入波形数据  
    figure(2) %绘制波形图
    plot(t*1000,s*1000,'color','red'); %图中时间单位转换为“毫秒”，电压单位转换为“毫伏”
    grid off;
    axis('tight')
    xlim([-2,11]*10^-4*1000);
    ylabel('电压 (mV)','fontsize',15)
    set(gca,'fontsize',15)
    xlabel('时间 (ms)','fontsize',15)
    
    
    wavename='cmor3-3'; % 选择时频分析用的小波
    totalscal=2^9;      % 小波分解尺度
    Fc=centfrq(wavename); % 小波的中心频率
    c=2*Fc*totalscal;
    scals=c./(1:totalscal);
    f=scal2frq(scals,wavename,1/fs); % 将尺度转换为频率
    coefs=cwt(s,scals,wavename); % 求连续小波系数
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
    ylabel('电压 (mV)','fontsize',15)
    set(gca,'fontsize',15)
    axes(a3);
    load('MyColormaps','mycmap')
    contourf(t*1000,f,abs(coefs),4);
    set(h1,'Colormap',mycmap)
    set(gca,'YDir','normal')
    ylim([30,250])
    xlim([-2,11]*10^-4*1000);
    xlabel('时间 (ms)','fontsize',15)
    set(gca,'fontsize',15)
    ch=colorbar;
    set(ch,'Location','East')
    axes(a2);
    [f,YY]=my_fft(s,fs);
    plot(YY/max(YY),f/1000,'linewidth',2);
    set(gca,'XDir','reverse')
    set(gca,'fontsize',15)
    ylabel('频率 (kHz)','fontsize',15)
    xlabel('正则化|Y|','fontsize',15)
    axis('tight')
    %xlim([0,15]*10^-3)
    ylim([30,250])
    
end
function [f,YY]=my_fft(y,Fs) %该函数对信号做fft变换得到频谱图
    L=length(y);
    NFFT = 2^nextpow2(L); % Next power of 2 from length of y
    Y = fft(y,NFFT)/L;
    f = Fs/2*linspace(0,1,NFFT/2);
    % Plot single-sided amplitude spectrum.
    YY=2*abs(Y(1:NFFT/2));
end  

