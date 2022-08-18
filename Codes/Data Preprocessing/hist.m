%%
MAP_TOT_UCI = zeros(1,300000);
for i = 1:length(ABP_GRND)
    ABP_PP = ABP_GRND(:,i);
    MAP = mean(ABP_PP);
    MAP_TOT_UCI(1,i) = MAP;
end
MAP = MAP_TOT_UCI(1,1:length(ABP));
%%
BP_TOT_UCI = zeros(1,300000);
for i = 1:length(ABP_GRND)
    BP_PP = SBP(1,i)-DBP(1,i);
    if BP_PP > 115
        % disp(BP_PP);
        disp(i);
    end
    BP_TOT_UCI(1,i) = BP_PP;
end
BP = BP_TOT_UCI(1,1:length(ABP_GRND));
%%
MAP_Max = max(SBP);
MAP_Min = min(SBP);
MAP_Mean = mean(SBP);
MAP_STD = std(SBP);
disp(MAP_Max)
disp(MAP_Min)
disp(MAP_Mean)
disp(MAP_STD)
%% Histogram
figure;
sgtitle('Histogram of Train Dataset before Data Cleansing','Color','blue','Fontsize',20);
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
sgtitle('Box Plot of Train Dataset before Data Cleansing','Color','blue','Fontsize',20);
subplot(1,3,1);
boxplot(SBP,'Labels','Systolic Blood Pressure (SBP)');
ylabel('Blood Pressure','Fontsize',14);
subplot(1,3,2);
boxplot(DBP,'Labels','Diastolic Blood Pressure (DBP)');
subplot(1,3,3);
boxplot(MAP,'Labels','Mean Arterial Pressure (MAP)');