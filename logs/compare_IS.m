clc
clear all
close all
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
%fname = 'BigGAN_I128_hdf5_seed0_Gch64_Dch64_bs256_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitxavier_Dinitxavier_Gshared_alex0'


%% Get dir
s = {};
d = dir();
j = 1;
for i = 1:length(d)
    if any(strfind(d(i).name,'I128')) && any(strfind(d(i).name,'.jsonl')) %&& any(strfind(d(i).name,'LARS'))
        s = [s; d(i).name];
    end
end

cc = jet(length(s))
%cc = [[1.0, 0.0,0.0]; [0.0, 0.6,0.0]; [0.0, 0.0,1.0]];
legend1 = {};
legend2 = {}
j = 1;
for i = 1:length(s)
    fname = s{i,1};
    %% Check if the Inception metrics log exists, and if so, plot it
    [itr, IS, FID] = process_inception_log(fname(1:end - 10), 'log.jsonl');
    s{i,2} = itr;
    s{i,3} = IS;
    s{i,4} = FID;
    if max(IS) > 8%11.5 % 8
        legend1 = [legend1; fname];
        figure(1)
        plot(itr, IS, 'color', cc(i,:),'linewidth',2)
        hold on;
        xlabel('itr'); ylabel('IS');
    %end
    %if ~(any(FID>500))
        legend2 = [legend2; fname];
        figure(2)
        plot(itr, FID, 'color', cc(i,:),'linewidth',2)
        hold on;
        xlabel('itr'); ylabel('FID');
        j = j + 1;
    end
   
end
%legend1{3} = 'Using Alex''s Generator'
%legend1{end} = 
%%
%legend1 = {'My previous best run', 'Alex''s Generator, no attention or hier', 'Lars Mescheder''s R1GP model and training code, my boilerplate'}
%legend2 = legend1;
%
figure(1)
legend(legend1)
figure(2)
legend(legend2)
%legend(s(:,1))
%     figure; subplot(2,1,1);
%     plot(itr,IS)%, 'color', [1.0, 0.0,0.0])
%     xlabel('itr'); ylabel('IS');
%     subplot(2,1,2);
%     plot(itr,FID)%, 'color', [0.0,1.0,0.0])
%     xlabel('itr'); ylabel('FID');