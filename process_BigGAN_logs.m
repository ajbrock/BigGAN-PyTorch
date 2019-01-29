clc
clear all
%close all
fclose all;
fname = 'BigGAN_I128_Gch64_Dch64_bs128_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Gattn0_Dattn0_seed14';
fname = 'BigGAN_I128_Gch64_Dch64_bs128_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Gattn0_Dattn0_seed128';
fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Gattn0_Dattn0_seed256';
fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Gattn0_Dattn0_seed22';
fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs2_nDa1_nGa1_Glr5.0e-05_Dlr2.0e-04_Gnlinplace_relu_Dnlinplace_relu_GinitN02_DinitN02_Gattn0_Dattn0_seed0_cr';
%fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs2_nDa1_nGa1_Glr5.0e-05_Dlr2.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn0_Dattn0_seed0_cr'; % Currently our best run on I128?
%fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs2_nDa1_nGa1_Glr5.0e-05_Dlr2.0e-04_Gnlinplace_relu_Dnlinplace_relu_GinitN02_DinitN02_Gattn0_Dattn0_seed0_cr'; % Not doing as well as the above run?
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn64_Dattn64_seed644_cr_ema';
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn65_Dattn64_seed644_cr_ema'; % Same as above but gattn65 (no diff really)
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn66_Dattn64_seed644_ema'; % This one seemd to not go well!
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn66_Dattn64_seed644_cr_ema'; % Same as above but cr
%fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs5_nDa1_nGa1_Glr2.0e-04_Dlr2.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn0_Dattn0_seed22' % my old_model run
%%
fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn0_Dattn0_seed27' % noattn_c run
%%
%fname = 'BigGAN_I128_Gch64_Dch64_bs256_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_GSVD_SN_DSVD_SN_Gattn0_Dattn0_seed2_cr_Gshared' % SVDSN run
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn64_Dattn64_seed0_ema' % a C10 run that works
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_SN_DSVD_SN_Gattn0_Dattn0_seed2_cr_Gshared' % My SVD_SN run with SVDSN on the output conv
%fname = 'BigGAN_C100_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_SN_DSVD_SN_Gattn0_Dattn0_seed2_cr_Gshared' % My C100 SVD_SN run with SVDSN on the output conv
%fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVDrange_0.5_1.0_DSVDrange_0.5_1.0_Gattn0_Dattn0_seed2_cr_Gshared'
% fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_DSVD_Gattn0_Dattn0_seed0_cr_Gshared_ema' % An EMA run which performed about as expected
% fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVDrange_0.8_1.0_DSVDrange_0.8_1.0_Gattn0_Dattn0_seed2_cr_Gshared' % a range run with a tighter range
% fname = 'BigGAN_C10_Gch64_Dch64_bs64_nDs1_nDa1_nGa1_Glr1.0e-05_Dlr4.0e-05_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_GSVD_SN_DSVD_SN_Gattn0_Dattn0_nonorm_seed2_Gshared' % my nonorm run with the LR decreased
% 
% % VAE runs
% fname = 'BigVAE_I128_DCT_Gch64_Dch64_bs256_Glr2.0e-05_Dlr2.0e-05_GB0.500_DB0.500_Gnlir_Dnlir_Ginitxavier_Dinitxavier_seed2_cr_Gshared_dct' % run without DCT coefficient scaling
% fname = 'BigVAE_I128_DCT_Gch64_Dch64_bs256_Glr2.0e-05_Dlr2.0e-05_GB0.500_DB0.500_Gnlir_Dnlir_Ginitxavier_Dinitxavier_seed42_cr_Gshared_dct' % Run with DCT coeff scaling
% 
% % Lars runs
%fname = 'BigLARS_I128_Gch64_Dch64_bs64_nDs2_Glr5.0e-05_Dlr2.0e-04_Gnlir_Dnlir_Ginitxavier_Dinitxavier_seed2'

%% BPT runs
%fname = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs64_nDs5_Glr2.0e-04_Dlr2.0e-04_GBB0.900_DBB0.900_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gno_SNGAN'
%fname = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_nDs5_Glr2.0e-04_Dlr2.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gshared_BPT'
%fname = 'BigGAN_I128_hdf5_seed2_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex2'
fname = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex0'
fname = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex0A'
fname = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlrelu_Dnlrelu_Ginitxavier_Dinitxavier_Gshared_jan27' % My run that should mimic alex0A to an extent
%% G_loss, D_loss
if exist(sprintf('%s_%s',fname, 'G_loss.log'))
    gl = process_log(fname, 'G_loss.log');    
    dlr = process_log(fname, 'D_loss_real.log');    
    dlf = process_log(fname, 'D_loss_fake.log');    
    fclose all;
    %% Plot with denoised
   % close all
    %x = 1:(i-1);
    p1 = figure;
    hold on;
    %p1.Color(4) = 0.2;
    %dlr2 = wden(dlr,'sqtwolog','s','mln',2,'sym4');
    thresh = 2;
    plot(1:length(gl),gl, 'color', [1.0, 0.7,0.7], 'HandleVisibility','off')
    plot(1:length(dlf),dlf, 'color', [0.7,1.0,0.7], 'HandleVisibility','off')
    plot(1:length(dlr),dlr, 'color', [0.7,0.7,1.0], 'HandleVisibility','off')
    plot(1:length(gl), wden(gl,'sqtwolog','s','mln', thresh, 'sym4'), 'Color', [1, 0, 0]);
    plot(1:length(dlf), wden(dlf,'sqtwolog','s','mln', thresh, 'sym4'), 'Color', [0, 1, 0]);
    plot(1:length(dlr), wden(dlr,'sqtwolog','s','mln', thresh, 'sym4'), 'Color', [0, 0, 1]);
    
    %plot(1:length(dlf), dlf, 1:length(gl), gl);%, x,gl);
    legend('gl', 'dlf', 'dlr');
    %axis([0,i * 1.5, -5,5])
end
%% Recon and KL?
if exist(sprintf('%s_%s',fname, 'kl.log'))
    kl = process_log(fname, 'kl.log');    
    recon = process_log(fname, 'recon.log');   
    %% Plot with denoised
    p1 = figure;
    hold on;
    thresh = 2;
    plot(1:length(kl),kl, 'color', [1.0, 0.7,0.7], 'HandleVisibility','off')
    plot(1:length(recon), recon, 'color', [0.7,1.0,0.7], 'HandleVisibility','off')    
    plot(1:length(kl), wden(kl,'sqtwolog','s','mln', thresh, 'sym4'), 'Color', [1, 0, 0]);
    plot(1:length(recon), wden(recon,'sqtwolog','s','mln', thresh, 'sym4'), 'Color', [0, 1, 0]);
    legend('KL', 'MSE');
    %axis([0,i * 1.5, -5,5])
end
if exist(sprintf('%s_%s',fname, 'reg.log'))
    reg = process_log(fname, 'reg.log');    
    fclose all;
    %% Plot with denoised
    p1 = figure;
    hold on;
    thresh = 2;
    plot(1:length(reg),reg, 'color', [1.0, 0.7,0.7], 'HandleVisibility','off')  
    plot(1:length(reg), wden(reg,'sqtwolog','s','mln', thresh, 'sym4'), 'Color', [1, 0, 0]);
    legend('regularization');
    %axis([0,i * 1.5, -5,5])
end

%% Check if the Inception metrics log exists, and if so, plot it
if exist(sprintf('%s_%s',fname, 'log.jsonl'))
    [itr, IS, FID] = process_inception_log(fname, 'log.jsonl');
    figure; subplot(2,1,1);
    plot(itr,IS)%, 'color', [1.0, 0.0,0.0])
    xlabel('itr'); ylabel('IS');
    subplot(2,1,2);
    plot(itr,FID)%, 'color', [0.0,1.0,0.0])
    xlabel('itr'); ylabel('FID');
end