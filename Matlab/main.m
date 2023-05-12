clear all
clc
addpath("..\");
%%
fileName = '..\test data\»¬Ê¯·Û1.60mm-1 12181014.xls';
[t, x] = ImageDataUtil.readData(fileName);
refX = x(:, 1);
sampX = x(:, 2);
d = 1.6;

ihht1 = ImprovedHhtFilter(t, refX);
ihht2 = ImprovedHhtFilter(t, sampX);
refXd = ihht1.apply();
sampXd = ihht2.apply();

sp = SampleProperties(t, refX, t, sampX, d, true);
refract = sp.calSampleRefraction();
absorp = sp.calSampleAbsorption();
[f, refract] = FrequencyAnalysisUtil.selectRangeOfFrequency(sp.f, refract, 0.1, 3);
[~, absorp] = FrequencyAnalysisUtil.selectRangeOfFrequency(sp.f, absorp, 0.1, 3);

sp1 = SampleProperties(t, refXd, t, sampXd, d, true);
refract1 = sp1.calSampleRefraction();
absorp1 = sp1.calSampleAbsorption();
[f1, refract1] = FrequencyAnalysisUtil.selectRangeOfFrequency(sp1.f, refract1, 0.1, 3);
[~, absorp1] = FrequencyAnalysisUtil.selectRangeOfFrequency(sp1.f, absorp1, 0.1, 3);

sp2 = SampleProperties(t, refX, t, sampX, d, true, true);
refract2 = sp2.calSampleRefraction();
absorp2 = sp2.calSampleAbsorption();
[f2, refract2] = FrequencyAnalysisUtil.selectRangeOfFrequency(sp2.f, refract2, 0.1, 3);
[~, absorp2] = FrequencyAnalysisUtil.selectRangeOfFrequency(sp2.f, absorp2, 0.1, 3);
%%
figure
subplot(2, 1, 1)
plot(f, refract)
title('Refractive index')
xlabel('Frequency (THz)')

subplot(2, 1, 2)
plot(f, absorp)
title('Absorption rate')
xlabel('Frequency (THz)')
ylabel('{cm^{-1}}')

figure
subplot(2, 1, 1)
plot(f1, refract1)
title('Refractive index without reflection peaks')
xlabel('Frequency (THz)')

subplot(2, 1, 2)
plot(f1, absorp1)
title('Absorption rate without reflection peaks')
xlabel('Frequency (THz)')
ylabel('{cm^{-1}}')

figure
subplot(2, 1, 1)
plot(f2, refract2)
title('Refractive index after deconvolution')
xlabel('Frequency (THz)')

subplot(2, 1, 2)
plot(f2, absorp2)
title('Absorption rate after deconvolution')
xlabel('Frequency (THz)')
ylabel('{cm^{-1}}')