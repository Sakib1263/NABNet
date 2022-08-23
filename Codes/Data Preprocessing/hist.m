%%
clear;
clc;
ABP_Part_1 = load('UCI_Dataset_Part_2_Preprocessed.mat','ABP_GRND');
ABP_Part_1 = ABP_Part_1.ABP_GRND;
ABP_Part_2 = load('UCI_Dataset_Part_3_Preprocessed.mat','ABP_GRND');
ABP_Part_2 = ABP_Part_2.ABP_GRND;
ABP_Part_3 = load('UCI_Dataset_Part_4_Preprocessed.mat','ABP_GRND');
ABP_Part_3 = ABP_Part_3.ABP_GRND;
% ABP_Part_4 = load('UCI_Dataset_Part_4_Preprocessed.mat','ABP_GRND');
% ABP_Part_4 = ABP_Part_4.ABP_GRND;
ABP_GRND = cat(2,ABP_Part_1,ABP_Part_2,ABP_Part_3);
%%
SBP_TOT_UCI = zeros(1,300000);
DBP_TOT_UCI = zeros(1,300000);
MAP_TOT_UCI = zeros(1,300000);
for i = 1:length(ABP_GRND)
    ABP_PP = ABP_GRND(:,i);
    SBP = max(ABP_PP);
    DBP = min(ABP_PP);
    MAP = mean(ABP_PP);
    SBP_TOT_UCI(1,i) = SBP;
    DBP_TOT_UCI(1,i) = DBP;
    MAP_TOT_UCI(1,i) = MAP;
end
SBP = SBP_TOT_UCI(1,1:length(ABP_GRND));
DBP = DBP_TOT_UCI(1,1:length(ABP_GRND));
MAP = MAP_TOT_UCI(1,1:length(ABP_GRND));
%%
% BP_TOT_UCI = zeros(1,300000);
% for i = 1:length(ABP_GRND)
%     BP_PP = SBP(1,i)-DBP(1,i);
%     if BP_PP > 115
%         % disp(BP_PP);
%         % disp(i);
%     end
%     BP_TOT_UCI(1,i) = BP_PP;
% end
% BP = BP_TOT_UCI(1,1:length(ABP_GRND));
%%
SBP_Max = max(SBP);
SBP_Min = min(SBP);
SBP_Mean = mean(SBP);
SBP_STD = std(SBP);
disp(SBP_Max)
disp(SBP_Min)
disp(SBP_Mean)
disp(SBP_STD)
%
DBP_Max = max(DBP);
DBP_Min = min(DBP);
DBP_Mean = mean(DBP);
DBP_STD = std(DBP);
disp(DBP_Max)
disp(DBP_Min)
disp(DBP_Mean)
disp(DBP_STD)
%
MAP_Max = max(MAP);
MAP_Min = min(MAP);
MAP_Mean = mean(MAP);
MAP_STD = std(MAP);
disp(MAP_Max)
disp(MAP_Min)
disp(MAP_Mean)
disp(MAP_STD)
%% Histogram
figure;
sgtitle('Histogram of Preprocessed UCI Dataset (Part 4)','Color','blue','Fontsize',20);
subplot(3,1,1);
X = histogram(SBP,1000);
xlabel('Systolic Blood Pressure (SBP) - Bins','Fontsize',14);
ylabel('Number of Samples','Fontsize',14);
subplot(3,1,2);
Y = histogram(DBP,1000);
xlabel('Diastolic Blood Pressure (DBP) - Bins','Fontsize',14);
ylabel('Number of Samples','Fontsize',14);
subplot(3,1,3);
Z = histogram(MAP,1000);
xlabel('Mean Arterial Pressure (MAP) - Bins','Fontsize',14);
ylabel('Number of Samples','Fontsize',14);
%% Box-Plot
figure;
sgtitle('Box Plot of Preprocessed UCI Dataset (Part 4)','Color','blue','Fontsize',20);
subplot(1,3,1);
boxplot(SBP,'Labels','Systolic Blood Pressure (SBP)');
ylabel('Blood Pressure','Fontsize',14);
subplot(1,3,2);
boxplot(DBP,'Labels','Diastolic Blood Pressure (DBP)');
subplot(1,3,3);
boxplot(MAP,'Labels','Mean Arterial Pressure (MAP)');
%%
clear;
clc;
ABP_GT = load('UCI_Dataset_Part_4_Preprocessed.mat','ABP_GRND');
ABP_GT = ABP_GT.ABP_GRND;
PPG_GT = load('UCI_Dataset_Part_4_Preprocessed.mat','PPG');
PPG_GT = PPG_GT.PPG;
VPG_GT = load('UCI_Dataset_Part_4_Preprocessed.mat','VPG');
VPG_GT = VPG_GT.VPG;
APG_GT = load('UCI_Dataset_Part_4_Preprocessed.mat','APG');
APG_GT = APG_GT.APG;
ECG_GT = load('UCI_Dataset_Part_4_Preprocessed.mat','ECG');
ECG_GT = ECG_GT.ECG;
ABP_Pred = squeeze(h5read('ABP_Estimated_Fold_1.h5','/ABP'));
%%
sig_num = sig_num-10;
A = PPG_GT(:,sig_num);
B = ECG_GT(:,sig_num);
C = ABP_GT(:,sig_num);
D = ABP_Pred(:,sig_num);
figure;
subplot(3,1,1);
plot(A,'LineWidth',2);
axis([0 1024 0 1])
title('PPG','FontSize',16);
subplot(3,1,2);
plot(B,'LineWidth',2);
axis([0 1024 0 1])
title('ECG','FontSize',16);
subplot(3,1,3);
hold on
plot(C,'LineWidth',2);
plot(D,'LineWidth',2);
hold off
axis([0 1024 min(C)-5 max(C)+5])
title('ABP','FontSize',16);
legend 'ABP GT' 'ABP Pred';
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% sgtitle('');
%%
clear;
clc;
ABP_GT = load('UCI_Dataset_Part_4_Preprocessed.mat','ABP_GRND');
ABP_GT = ABP_GT.ABP_GRND;
ABP_Pred = squeeze(h5read('ABP_Estimated_Fold_1.h5','/ABP'));
SBP_Error = [];
DBP_Error = [];
MAP_Error = [];
length_ABP = length(ABP_GT);
for i = 1:length_ABP
    ABP_GT_temp = ABP_GT(:,i);
    ABP_Pred_temp = ABP_Pred(:,i);
    SBP_Error_temp = abs(max(ABP_GT_temp) - max(ABP_Pred_temp));
    DBP_Error_temp = abs(min(ABP_GT_temp) - min(ABP_Pred_temp));
    MAP_Error_temp = abs(mean(ABP_GT_temp) - mean(ABP_Pred_temp));
    SBP_Error = [SBP_Error SBP_Error_temp];
    DBP_Error = [DBP_Error DBP_Error_temp];
    MAP_Error = [MAP_Error MAP_Error_temp];
end
SBP_Error = mean(SBP_Error);
DBP_Error = mean(DBP_Error);
MAP_Error = mean(MAP_Error);