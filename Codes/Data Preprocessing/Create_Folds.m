Data_Length = length(PPG);
signal_length = 1024;
%% Prepare Test Data
Test_Data_Length = Data_Length / 5; %20% Data for Testing, 5Fold Cross Validation
%%
PPG_Test = PPG(:,Test_Data_Length*4+1:Data_Length);
ABP_Test = ABP(:,Test_Data_Length*4+1:Data_Length);
ABP_GRND_Test = ABP_GRND(:,Test_Data_Length*4+1:Data_Length);
ECG_Test = ECG(:,Test_Data_Length*4+1:Data_Length);
SBP_Test = SBP(:,Test_Data_Length*4+1:Data_Length);
DBP_Test = DBP(:,Test_Data_Length*4+1:Data_Length);
%
save('Folds/Test_Data.mat','PPG_Test','ABP_Test','ABP_GRND_Test','ECG_Test','SBP_Test','DBP_Test','-v7.3');
%
h5create('Folds/Test_Data.h5','/PPG',[signal_length length(PPG_Test)]);
h5create('Folds/Test_Data.h5','/ABP',[signal_length length(PPG_Test)]);
h5create('Folds/Test_Data.h5','/ECG',[signal_length length(PPG_Test)]);
h5create('Folds/Test_Data.h5','/SBP',[1 length(PPG_Test)]);
h5create('Folds/Test_Data.h5','/DBP',[1 length(PPG_Test)]);
h5create('Folds/Test_Data.h5','/ABP_GRND',[signal_length length(PPG_Test)]);
h5write('Folds/Test_Data.h5','/PPG',PPG_Test);
h5write('Folds/Test_Data.h5','/ABP',ABP_Test);
h5write('Folds/Test_Data.h5','/ECG',ECG_Test);
h5write('Folds/Test_Data.h5','/SBP',SBP_Test);
h5write('Folds/Test_Data.h5','/DBP',DBP_Test);
h5write('Folds/Test_Data.h5','/ABP_GRND',ABP_GRND_Test);
%% Rest Data (Train + Validation)
PPG_Rest = PPG(:,1:Test_Data_Length*4);
ABP_Rest = ABP(:,1:Test_Data_Length*4);
ABP_GRND_Rest = ABP_GRND(:,1:Test_Data_Length*4);
ECG_Rest = ECG(:,1:Test_Data_Length*4);
SBP_Rest = SBP(:,1:Test_Data_Length*4);
DBP_Rest = DBP(:,1:Test_Data_Length*4);
%
Data_Length_Rest = length(PPG_Rest);
Rest_Fifth = Data_Length_Rest / 5; %20% Data for Testing, 5Fold Cross Validation
%% Fold1
Train_PPG_Fold1 = PPG_Rest(:,Rest_Fifth+1:Rest_Fifth*5);
Train_ABP_Fold1 = ABP_Rest(:,Rest_Fifth+1:Rest_Fifth*5);
Train_ABP_GRND_Fold1 = ABP_GRND_Rest(:,Rest_Fifth+1:Rest_Fifth*5);
Train_ECG_Fold1 = ECG_Rest(:,Rest_Fifth+1:Rest_Fifth*5);
Train_SBP_Fold1 = SBP_Rest(:,Rest_Fifth+1:Rest_Fifth*5);
Train_DBP_Fold1 = DBP_Rest(:,Rest_Fifth+1:Rest_Fifth*5);
%
Val_PPG_Fold1 = PPG_Rest(:,1:Rest_Fifth);
Val_ABP_Fold1 = ABP_Rest(:,1:Rest_Fifth);
Val_ABP_GRND_Fold1 = ABP_GRND_Rest(:,1:Rest_Fifth);
Val_ECG_Fold1 = ECG_Rest(:,1:Rest_Fifth);
Val_SBP_Fold1 = SBP_Rest(:,1:Rest_Fifth);
Val_DBP_Fold1 = DBP_Rest(:,1:Rest_Fifth);

save('Folds/Fold 1/Train_Data_Fold1.mat','Train_PPG_Fold1','Train_ABP_Fold1','Train_ABP_GRND_Fold1','Train_ECG_Fold1','Train_SBP_Fold1','Train_DBP_Fold1','-v7.3');
save('Folds/Fold 1/Val_Data_Fold1.mat','Val_PPG_Fold1','Val_ABP_Fold1','Val_ABP_GRND_Fold1','Val_ECG_Fold1','Val_SBP_Fold1','Val_DBP_Fold1','-v7.3');
%
h5create('Folds/Fold 1/Train_Data_Fold1.h5','/PPG',[signal_length length(Train_PPG_Fold1)]);
h5create('Folds/Fold 1/Train_Data_Fold1.h5','/ABP',[signal_length length(Train_PPG_Fold1)]);
h5create('Folds/Fold 1/Train_Data_Fold1.h5','/ECG',[signal_length length(Train_PPG_Fold1)]);
h5create('Folds/Fold 1/Train_Data_Fold1.h5','/SBP',[1 length(Train_PPG_Fold1)]);
h5create('Folds/Fold 1/Train_Data_Fold1.h5','/DBP',[1 length(Train_PPG_Fold1)]);
h5create('Folds/Fold 1/Train_Data_Fold1.h5','/ABP_GRND',[signal_length length(Train_PPG_Fold1)]);
h5write('Folds/Fold 1/Train_Data_Fold1.h5','/PPG',Train_PPG_Fold1);
h5write('Folds/Fold 1/Train_Data_Fold1.h5','/ABP',Train_ABP_Fold1);
h5write('Folds/Fold 1/Train_Data_Fold1.h5','/ECG',Train_ECG_Fold1);
h5write('Folds/Fold 1/Train_Data_Fold1.h5','/SBP',Train_SBP_Fold1);
h5write('Folds/Fold 1/Train_Data_Fold1.h5','/DBP',Train_DBP_Fold1);
h5write('Folds/Fold 1/Train_Data_Fold1.h5','/ABP_GRND',Train_ABP_GRND_Fold1);
%
h5create('Folds/Fold 1/Val_Data_Fold1.h5','/PPG',[signal_length length(Val_PPG_Fold1)]);
h5create('Folds/Fold 1/Val_Data_Fold1.h5','/ABP',[signal_length length(Val_PPG_Fold1)]);
h5create('Folds/Fold 1/Val_Data_Fold1.h5','/ECG',[signal_length length(Val_PPG_Fold1)]);
h5create('Folds/Fold 1/Val_Data_Fold1.h5','/SBP',[1 length(Val_PPG_Fold1)]);
h5create('Folds/Fold 1/Val_Data_Fold1.h5','/DBP',[1 length(Val_PPG_Fold1)]);
h5create('Folds/Fold 1/Val_Data_Fold1.h5','/ABP_GRND',[signal_length length(Val_PPG_Fold1)]);
h5write('Folds/Fold 1/Val_Data_Fold1.h5','/PPG',Val_PPG_Fold1);
h5write('Folds/Fold 1/Val_Data_Fold1.h5','/ABP',Val_ABP_Fold1);
h5write('Folds/Fold 1/Val_Data_Fold1.h5','/ECG',Val_ECG_Fold1);
h5write('Folds/Fold 1/Val_Data_Fold1.h5','/SBP',Val_SBP_Fold1);
h5write('Folds/Fold 1/Val_Data_Fold1.h5','/DBP',Val_DBP_Fold1);
h5write('Folds/Fold 1/Val_Data_Fold1.h5','/ABP_GRND',Val_ABP_GRND_Fold1);
%% Fold2
Train_PPG_Fold2 = horzcat(PPG_Rest(:,1:Rest_Fifth), PPG_Rest(:,Rest_Fifth*2+1:Rest_Fifth*5));
Train_ABP_Fold2 = horzcat(ABP_Rest(:,1:Rest_Fifth), ABP_Rest(:,Rest_Fifth*2+1:Rest_Fifth*5));
Train_ABP_GRND_Fold2 = horzcat(ABP_GRND_Rest(:,1:Rest_Fifth), ABP_GRND_Rest(:,Rest_Fifth*2+1:Rest_Fifth*5));
Train_ECG_Fold2 = horzcat(ECG_Rest(:,1:Rest_Fifth), ECG_Rest(:,Rest_Fifth*2+1:Rest_Fifth*5));
Train_SBP_Fold2 = horzcat(SBP_Rest(:,1:Rest_Fifth), SBP_Rest(:,Rest_Fifth*2+1:Rest_Fifth*5));
Train_DBP_Fold2 = horzcat(DBP_Rest(:,1:Rest_Fifth), DBP_Rest(:,Rest_Fifth*2+1:Rest_Fifth*5));
%
Val_PPG_Fold2 = PPG_Rest(:,Rest_Fifth+1:Rest_Fifth*2);
Val_ABP_Fold2 = ABP_Rest(:,Rest_Fifth+1:Rest_Fifth*2);
Val_ABP_GRND_Fold2 = ABP_GRND_Rest(:,Rest_Fifth+1:Rest_Fifth*2);
Val_ECG_Fold2 = ECG_Rest(:,Rest_Fifth+1:Rest_Fifth*2);
Val_SBP_Fold2 = SBP_Rest(:,Rest_Fifth+1:Rest_Fifth*2);
Val_DBP_Fold2 = DBP_Rest(:,Rest_Fifth+1:Rest_Fifth*2);
%
save('Folds/Fold 2/Train_Data_Fold2.mat','Train_PPG_Fold2','Train_ABP_Fold2','Train_ABP_GRND_Fold2','Train_ECG_Fold2','Train_SBP_Fold2','Train_DBP_Fold2','-v7.3');
save('Folds/Fold 2/Val_Data_Fold2.mat','Val_PPG_Fold2','Val_ABP_Fold2','Val_ABP_GRND_Fold2','Val_ECG_Fold2','Val_SBP_Fold2','Val_DBP_Fold2','-v7.3');
%
h5create('Folds/Fold 2/Train_Data_Fold2.h5','/PPG',[signal_length length(Train_PPG_Fold2)]);
h5create('Folds/Fold 2/Train_Data_Fold2.h5','/ABP',[signal_length length(Train_PPG_Fold2)]);
h5create('Folds/Fold 2/Train_Data_Fold2.h5','/ECG',[signal_length length(Train_PPG_Fold2)]);
h5create('Folds/Fold 2/Train_Data_Fold2.h5','/SBP',[1 length(Train_PPG_Fold2)]);
h5create('Folds/Fold 2/Train_Data_Fold2.h5','/DBP',[1 length(Train_PPG_Fold2)]);
h5create('Folds/Fold 2/Train_Data_Fold2.h5','/ABP_GRND',[signal_length length(Train_PPG_Fold2)]);
h5write('Folds/Fold 2/Train_Data_Fold2.h5','/PPG',Train_PPG_Fold2);
h5write('Folds/Fold 2/Train_Data_Fold2.h5','/ABP',Train_ABP_Fold2);
h5write('Folds/Fold 2/Train_Data_Fold2.h5','/ECG',Train_ECG_Fold2);
h5write('Folds/Fold 2/Train_Data_Fold2.h5','/SBP',Train_SBP_Fold2);
h5write('Folds/Fold 2/Train_Data_Fold2.h5','/DBP',Train_DBP_Fold2);
h5write('Folds/Fold 2/Train_Data_Fold2.h5','/ABP_GRND',Train_ABP_GRND_Fold2);
%
h5create('Folds/Fold 2/Val_Data_Fold2.h5','/PPG',[signal_length length(Val_PPG_Fold2)]);
h5create('Folds/Fold 2/Val_Data_Fold2.h5','/ABP',[signal_length length(Val_PPG_Fold2)]);
h5create('Folds/Fold 2/Val_Data_Fold2.h5','/ECG',[signal_length length(Val_PPG_Fold2)]);
h5create('Folds/Fold 2/Val_Data_Fold2.h5','/SBP',[1 length(Val_PPG_Fold2)]);
h5create('Folds/Fold 2/Val_Data_Fold2.h5','/DBP',[1 length(Val_PPG_Fold2)]);
h5create('Folds/Fold 2/Val_Data_Fold2.h5','/ABP_GRND',[signal_length length(Val_PPG_Fold2)]);
h5write('Folds/Fold 2/Val_Data_Fold2.h5','/PPG',Val_PPG_Fold2);
h5write('Folds/Fold 2/Val_Data_Fold2.h5','/ABP',Val_ABP_Fold2);
h5write('Folds/Fold 2/Val_Data_Fold2.h5','/ECG',Val_ECG_Fold2);
h5write('Folds/Fold 2/Val_Data_Fold2.h5','/SBP',Val_SBP_Fold2);
h5write('Folds/Fold 2/Val_Data_Fold2.h5','/DBP',Val_DBP_Fold2);
h5write('Folds/Fold 2/Val_Data_Fold2.h5','/ABP_GRND',Val_ABP_GRND_Fold2);
%% Fold3
Train_PPG_Fold3 = horzcat(PPG_Rest(:,1:Rest_Fifth*2), PPG_Rest(:,Rest_Fifth*3+1:Rest_Fifth*5));
Train_ABP_Fold3 = horzcat(ABP_Rest(:,1:Rest_Fifth*2), ABP_Rest(:,Rest_Fifth*3+1:Rest_Fifth*5));
Train_ABP_GRND_Fold3 = horzcat(ABP_GRND_Rest(:,1:Rest_Fifth*2), ABP_GRND_Rest(:,Rest_Fifth*3+1:Rest_Fifth*5));
Train_ECG_Fold3 = horzcat(ECG_Rest(:,1:Rest_Fifth*2), ECG_Rest(:,Rest_Fifth*3+1:Rest_Fifth*5));
Train_SBP_Fold3 = horzcat(SBP_Rest(:,1:Rest_Fifth*2), SBP_Rest(:,Rest_Fifth*3+1:Rest_Fifth*5));
Train_DBP_Fold3 = horzcat(DBP_Rest(:,1:Rest_Fifth*2), DBP_Rest(:,Rest_Fifth*3+1:Rest_Fifth*5));
%
Val_PPG_Fold3 = PPG_Rest(:,Rest_Fifth*2+1:Rest_Fifth*3);
Val_ABP_Fold3 = ABP_Rest(:,Rest_Fifth*2+1:Rest_Fifth*3);
Val_ABP_GRND_Fold3 = ABP_GRND_Rest(:,Rest_Fifth*2+1:Rest_Fifth*3);
Val_ECG_Fold3 = ECG_Rest(:,Rest_Fifth*2+1:Rest_Fifth*3);
Val_SBP_Fold3 = SBP_Rest(:,Rest_Fifth*2+1:Rest_Fifth*3);
Val_DBP_Fold3 = DBP_Rest(:,Rest_Fifth*2+1:Rest_Fifth*3);
%
save('Folds/Fold 3/Train_Data_Fold3.mat','Train_PPG_Fold3','Train_ABP_Fold3','Train_ABP_GRND_Fold3','Train_ECG_Fold3','Train_SBP_Fold3','Train_DBP_Fold3','-v7.3');
save('Folds/Fold 3/Val_Data_Fold3.mat','Val_PPG_Fold3','Val_ABP_Fold3','Val_ABP_GRND_Fold3','Val_ECG_Fold3','Val_SBP_Fold3','Val_DBP_Fold3','-v7.3');
%
h5create('Folds/Fold 3/Train_Data_Fold3.h5','/PPG',[signal_length length(Train_PPG_Fold3)]);
h5create('Folds/Fold 3/Train_Data_Fold3.h5','/ABP',[signal_length length(Train_PPG_Fold3)]);
h5create('Folds/Fold 3/Train_Data_Fold3.h5','/ECG',[signal_length length(Train_PPG_Fold3)]);
h5create('Folds/Fold 3/Train_Data_Fold3.h5','/SBP',[1 length(Train_PPG_Fold3)]);
h5create('Folds/Fold 3/Train_Data_Fold3.h5','/DBP',[1 length(Train_PPG_Fold3)]);
h5create('Folds/Fold 3/Train_Data_Fold3.h5','/ABP_GRND',[signal_length length(Train_PPG_Fold3)]);
h5write('Folds/Fold 3/Train_Data_Fold3.h5','/PPG',Train_PPG_Fold3);
h5write('Folds/Fold 3/Train_Data_Fold3.h5','/ABP',Train_ABP_Fold3);
h5write('Folds/Fold 3/Train_Data_Fold3.h5','/ECG',Train_ECG_Fold3);
h5write('Folds/Fold 3/Train_Data_Fold3.h5','/SBP',Train_SBP_Fold3);
h5write('Folds/Fold 3/Train_Data_Fold3.h5','/DBP',Train_DBP_Fold3);
h5write('Folds/Fold 3/Train_Data_Fold3.h5','/ABP_GRND',Train_ABP_GRND_Fold3);
%
h5create('Folds/Fold 3/Val_Data_Fold3.h5','/PPG',[signal_length length(Val_PPG_Fold3)]);
h5create('Folds/Fold 3/Val_Data_Fold3.h5','/ABP',[signal_length length(Val_PPG_Fold3)]);
h5create('Folds/Fold 3/Val_Data_Fold3.h5','/ECG',[signal_length length(Val_PPG_Fold3)]);
h5create('Folds/Fold 3/Val_Data_Fold3.h5','/SBP',[1 length(Val_PPG_Fold3)]);
h5create('Folds/Fold 3/Val_Data_Fold3.h5','/DBP',[1 length(Val_PPG_Fold3)]);
h5create('Folds/Fold 3/Val_Data_Fold3.h5','/ABP_GRND',[signal_length length(Val_PPG_Fold3)]);
h5write('Folds/Fold 3/Val_Data_Fold3.h5','/PPG',Val_PPG_Fold3);
h5write('Folds/Fold 3/Val_Data_Fold3.h5','/ABP',Val_ABP_Fold3);
h5write('Folds/Fold 3/Val_Data_Fold3.h5','/ECG',Val_ECG_Fold3);
h5write('Folds/Fold 3/Val_Data_Fold3.h5','/SBP',Val_SBP_Fold3);
h5write('Folds/Fold 3/Val_Data_Fold3.h5','/DBP',Val_DBP_Fold3);
h5write('Folds/Fold 3/Val_Data_Fold3.h5','/ABP_GRND',Val_ABP_GRND_Fold3);
%% Fold4
Train_PPG_Fold4 = horzcat(PPG_Rest(:,1:Rest_Fifth*3), PPG_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5));
Train_ABP_Fold4 = horzcat(ABP_Rest(:,1:Rest_Fifth*3), ABP_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5));
Train_ABP_GRND_Fold4 = horzcat(ABP_GRND_Rest(:,1:Rest_Fifth*3), ABP_GRND_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5));
Train_ECG_Fold4 = horzcat(ECG_Rest(:,1:Rest_Fifth*3), ECG_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5));
Train_SBP_Fold4 = horzcat(SBP_Rest(:,1:Rest_Fifth*3), SBP_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5));
Train_DBP_Fold4 = horzcat(DBP_Rest(:,1:Rest_Fifth*3), DBP_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5));
%
Val_PPG_Fold4 = PPG_Rest(:,Rest_Fifth*3+1:Rest_Fifth*4);
Val_ABP_Fold4 = ABP_Rest(:,Rest_Fifth*3+1:Rest_Fifth*4);
Val_ABP_GRND_Fold4 = ABP_GRND_Rest(:,Rest_Fifth*3+1:Rest_Fifth*4);
Val_ECG_Fold4 = ECG_Rest(:,Rest_Fifth*3+1:Rest_Fifth*4);
Val_SBP_Fold4 = SBP_Rest(:,Rest_Fifth*3+1:Rest_Fifth*4);
Val_DBP_Fold4 = DBP_Rest(:,Rest_Fifth*3+1:Rest_Fifth*4);

save('Folds/Fold 4/Train_Data_Fold4.mat','Train_PPG_Fold4','Train_ABP_Fold4','Train_ABP_GRND_Fold4','Train_ECG_Fold4','Train_SBP_Fold4','Train_DBP_Fold4','-v7.3');
save('Folds/Fold 4/Val_Data_Fold4.mat','Val_PPG_Fold4','Val_ABP_Fold4','Val_ABP_GRND_Fold4','Val_ECG_Fold4','Val_SBP_Fold4','Val_DBP_Fold4','-v7.3');
%
h5create('Folds/Fold 4/Train_Data_Fold4.h5','/PPG',[signal_length length(Train_PPG_Fold4)]);
h5create('Folds/Fold 4/Train_Data_Fold4.h5','/ABP',[signal_length length(Train_PPG_Fold4)]);
h5create('Folds/Fold 4/Train_Data_Fold4.h5','/ECG',[signal_length length(Train_PPG_Fold4)]);
h5create('Folds/Fold 4/Train_Data_Fold4.h5','/SBP',[1 length(Train_PPG_Fold4)]);
h5create('Folds/Fold 4/Train_Data_Fold4.h5','/DBP',[1 length(Train_PPG_Fold4)]);
h5create('Folds/Fold 4/Train_Data_Fold4.h5','/ABP_GRND',[signal_length length(Train_PPG_Fold4)]);
h5write('Folds/Fold 4/Train_Data_Fold4.h5','/PPG',Train_PPG_Fold4);
h5write('Folds/Fold 4/Train_Data_Fold4.h5','/ABP',Train_ABP_Fold4);
h5write('Folds/Fold 4/Train_Data_Fold4.h5','/ECG',Train_ECG_Fold4);
h5write('Folds/Fold 4/Train_Data_Fold4.h5','/SBP',Train_SBP_Fold4);
h5write('Folds/Fold 4/Train_Data_Fold4.h5','/DBP',Train_DBP_Fold4);
h5write('Folds/Fold 4/Train_Data_Fold4.h5','/ABP_GRND',Train_ABP_GRND_Fold4);
%
h5create('Folds/Fold 4/Val_Data_Fold4.h5','/PPG',[signal_length length(Val_PPG_Fold4)]);
h5create('Folds/Fold 4/Val_Data_Fold4.h5','/ABP',[signal_length length(Val_PPG_Fold4)]);
h5create('Folds/Fold 4/Val_Data_Fold4.h5','/ECG',[signal_length length(Val_PPG_Fold4)]);
h5create('Folds/Fold 4/Val_Data_Fold4.h5','/SBP',[1 length(Val_PPG_Fold4)]);
h5create('Folds/Fold 4/Val_Data_Fold4.h5','/DBP',[1 length(Val_PPG_Fold4)]);
h5create('Folds/Fold 4/Val_Data_Fold4.h5','/ABP_GRND',[signal_length length(Val_PPG_Fold4)]);
h5write('Folds/Fold 4/Val_Data_Fold4.h5','/PPG',Val_PPG_Fold4);
h5write('Folds/Fold 4/Val_Data_Fold4.h5','/ABP',Val_ABP_Fold4);
h5write('Folds/Fold 4/Val_Data_Fold4.h5','/ECG',Val_ECG_Fold4);
h5write('Folds/Fold 4/Val_Data_Fold4.h5','/SBP',Val_SBP_Fold4);
h5write('Folds/Fold 4/Val_Data_Fold4.h5','/DBP',Val_DBP_Fold4);
h5write('Folds/Fold 4/Val_Data_Fold4.h5','/ABP_GRND',Val_ABP_GRND_Fold4);
%% Fold5
Train_PPG_Fold5 = PPG_Rest(:,1:Rest_Fifth*4);
Train_ABP_Fold5 = ABP_Rest(:,1:Rest_Fifth*4);
Train_ABP_GRND_Fold5 = ABP_GRND_Rest(:,1:Rest_Fifth*4);
Train_ECG_Fold5 = ECG_Rest(:,1:Rest_Fifth*4);
Train_SBP_Fold5 = SBP_Rest(:,1:Rest_Fifth*4);
Train_DBP_Fold5 = DBP_Rest(:,1:Rest_Fifth*4);
%
Val_PPG_Fold5 = PPG_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5);
Val_ABP_Fold5 = ABP_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5);
Val_ABP_GRND_Fold5 = ABP_GRND_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5);
Val_ECG_Fold5 = ECG_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5);
Val_SBP_Fold5 = SBP_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5);
Val_DBP_Fold5 = DBP_Rest(:,Rest_Fifth*4+1:Rest_Fifth*5);

save('Folds/Fold 5/Train_Data_Fold5.mat','Train_PPG_Fold5','Train_ABP_Fold5','Train_ABP_GRND_Fold5','Train_ECG_Fold5','Train_SBP_Fold5','Train_DBP_Fold5','-v7.3');
save('Folds/Fold 5/Val_Data_Fold5.mat','Val_PPG_Fold5','Val_ABP_Fold5','Val_ABP_GRND_Fold5','Val_ECG_Fold5','Val_SBP_Fold5','Val_DBP_Fold5','-v7.3');
%
h5create('Folds/Fold 5/Train_Data_Fold5.h5','/PPG',[signal_length length(Train_PPG_Fold5)]);
h5create('Folds/Fold 5/Train_Data_Fold5.h5','/ABP',[signal_length length(Train_PPG_Fold5)]);
h5create('Folds/Fold 5/Train_Data_Fold5.h5','/ECG',[signal_length length(Train_PPG_Fold5)]);
h5create('Folds/Fold 5/Train_Data_Fold5.h5','/SBP',[1 length(Train_PPG_Fold5)]);
h5create('Folds/Fold 5/Train_Data_Fold5.h5','/DBP',[1 length(Train_PPG_Fold5)]);
h5create('Folds/Fold 5/Train_Data_Fold5.h5','/ABP_GRND',[signal_length length(Train_PPG_Fold5)]);
h5write('Folds/Fold 5/Train_Data_Fold5.h5','/PPG',Train_PPG_Fold5);
h5write('Folds/Fold 5/Train_Data_Fold5.h5','/ABP',Train_ABP_Fold5);
h5write('Folds/Fold 5/Train_Data_Fold5.h5','/ECG',Train_ECG_Fold5);
h5write('Folds/Fold 5/Train_Data_Fold5.h5','/SBP',Train_SBP_Fold5);
h5write('Folds/Fold 5/Train_Data_Fold5.h5','/DBP',Train_DBP_Fold5);
h5write('Folds/Fold 5/Train_Data_Fold5.h5','/ABP_GRND',Train_ABP_GRND_Fold5);
%
h5create('Folds/Fold 5/Val_Data_Fold5.h5','/PPG',[signal_length length(Val_PPG_Fold5)]);
h5create('Folds/Fold 5/Val_Data_Fold5.h5','/ABP',[signal_length length(Val_PPG_Fold5)]);
h5create('Folds/Fold 5/Val_Data_Fold5.h5','/ECG',[signal_length length(Val_PPG_Fold5)]);
h5create('Folds/Fold 5/Val_Data_Fold5.h5','/SBP',[1 length(Val_PPG_Fold5)]);
h5create('Folds/Fold 5/Val_Data_Fold5.h5','/DBP',[1 length(Val_PPG_Fold5)]);
h5create('Folds/Fold 5/Val_Data_Fold5.h5','/ABP_GRND',[signal_length length(Val_PPG_Fold5)]);
h5write('Folds/Fold 5/Val_Data_Fold5.h5','/PPG',Val_PPG_Fold5);
h5write('Folds/Fold 5/Val_Data_Fold5.h5','/ABP',Val_ABP_Fold5);
h5write('Folds/Fold 5/Val_Data_Fold5.h5','/ECG',Val_ECG_Fold5);
h5write('Folds/Fold 5/Val_Data_Fold5.h5','/SBP',Val_SBP_Fold5);
h5write('Folds/Fold 5/Val_Data_Fold5.h5','/DBP',Val_DBP_Fold5);
h5write('Folds/Fold 5/Val_Data_Fold5.h5','/ABP_GRND',Val_ABP_GRND_Fold5);