%% Split UCI Train-Test Data
clear;
load('UCI Dataset/Part_1.mat');
load('UCI Dataset/Part_2.mat');
load('UCI Dataset/Part_3.mat');
load('UCI Dataset/Part_4.mat');
UCI_Train_Dataset = horzcat(Part_1, Part_2, Part_3);
UCI_Test_Dataset = Part_4;
%% Prepare UCI Train and Test Datasets
warning('off');
% Input Signal Length should be at least 60 samples more than the Output, or more
input_signal_length = 1100;
output_signal_length = 1024;
% Make or Set Destination Directory
train_path_mat = sprintf('UCI_Train_Dataset.mat');
train_path_hdf5 = sprintf('UCI_Train_Dataset.h5');
test_path_mat = sprintf('UCI_Test_Dataset.mat');
test_path_hdf5 = sprintf('UCI_Test_Dataset.h5');
%
for c = 1:2
    if c == 1
        dataset_size = length(UCI_Train_Dataset);
    elseif c == 2
        dataset_size = length(UCI_Test_Dataset);
    end
    % No resampling required.
    % The Data for this dataset has been collected at 125Hz Sampling Frequency by Default
    counter = 0;
    bad_signal_count = 0;
    %
    PPG_TOT_UCI = zeros(output_signal_length,300000);
    VPG_TOT_UCI = zeros(output_signal_length,300000);
    APG_TOT_UCI = zeros(output_signal_length,300000);
    ECG_TOT_UCI = zeros(output_signal_length,300000);
    ABP_TOT_UCI = zeros(output_signal_length,300000);
    ABP_RNorm_UCI = zeros(output_signal_length,300000);
    SBP_TOT_UCI = zeros(1,300000);
    DBP_TOT_UCI = zeros(1,300000);
    
    for i = 1:dataset_size
        if c == 1
            cell = UCI_Train_Dataset{1,i};
        elseif c == 2
            cell = UCI_Test_Dataset{1,i};
        end
        cell_length = length(cell);
        if cell_length < input_signal_length
            continue;
        end
        PPG = cell(1,:);
        ABP = cell(2,:);
        ECG = cell(3,:);
        sample_num = floor(cell_length/input_signal_length);
        L_temp = sample_num*input_signal_length;
        PPG = PPG(1:L_temp);
        ABP = ABP(1:L_temp);
        ECG = ECG(1:L_temp);
        for ii = 0:(sample_num-1)
            counter = counter+1;
            Iter_Left = dataset_size - i;
            %
            PPG_PP = normalize(Fix_Baseline_Drift(PPG(input_signal_length*ii+1:input_signal_length*ii+input_signal_length)),'range');
            %
            ABP_PP1 = ABP(input_signal_length*ii+1:input_signal_length*ii+input_signal_length);
            ABP_PP2 = Fix_Baseline_Drift(ABP(input_signal_length*ii+1:input_signal_length*ii+input_signal_length));
            ABP_PP1_AMP = max(ABP_PP1) - min(ABP_PP1);
            ABP_PP2_AMP = max(ABP_PP2) - min(ABP_PP2);
            ABP_PP = (ABP_PP2*(ABP_PP1_AMP/ABP_PP2_AMP))+min(ABP_PP1);
            ABP_PP_RNorm = normalize(ABP_PP,'range');
            %
            ECG_PP = normalize(Fix_Baseline_Drift(ECG(input_signal_length*ii+1:input_signal_length*ii+input_signal_length)),'range');
            %
            [PPG_PP,VPG_PP,APG_PP,delay] = PPG_diff(PPG_PP,input_signal_length,output_signal_length);
            ABP_PP = crop_signal_delay(ABP_PP,output_signal_length,delay);
            ABP_PP_RNorm = crop_signal_delay(ABP_PP_RNorm,output_signal_length,delay);
            ECG_PP = crop_signal_delay(ECG_PP,output_signal_length,delay);
            %
            Decision = Remove_Bad_Signals(PPG_PP,ABP_PP,output_signal_length);
            TF = mean(isnan(ABP_PP));
            SBP = max(ABP_PP);
            DBP = min(ABP_PP);
            BP_DIFF = SBP - DBP;
            if (TF > 0)||((SBP > 190)||(SBP < 80)||(DBP > 120)||(DBP < 50)||(BP_DIFF < 20)||(BP_DIFF > 120))||(Decision == 0)
                bad_signal_count = bad_signal_count + 1;
                counter = counter - 1;
                continue
            else
                PPG_TOT_UCI(:,counter) = PPG_PP';
                VPG_TOT_UCI(:,counter) = VPG_PP';
                APG_TOT_UCI(:,counter) = APG_PP';
                ECG_TOT_UCI(:,counter) = ECG_PP';
                ABP_TOT_UCI(:,counter) = ABP_PP';
                ABP_RNorm_UCI(:,counter) = ABP_PP_RNorm';
                SBP_TOT_UCI(1,counter) = SBP;
                DBP_TOT_UCI(1,counter) = DBP;
            end
            fprintf('Samples Collected = %d, Iteration Left = %d, Bad Signal Count = %d\n',counter,Iter_Left,bad_signal_count);
        end
    end

    PPG = PPG_TOT_UCI(:,1:counter);
    VPG = VPG_TOT_UCI(:,1:counter);
    APG = APG_TOT_UCI(:,1:counter);
    ECG = ECG_TOT_UCI(:,1:counter);
    ABP = ABP_TOT_UCI(:,1:counter);
    ABP_RNorm = ABP_RNorm_UCI(:,1:counter);
    SBP = SBP_TOT_UCI(1,1:counter);
    DBP = DBP_TOT_UCI(1,1:counter);
    %
    ABP_GRND = ABP;
    ABP_MAX = max(max(ABP));
    ABP_MIN = min(min(ABP));
    ABP_AMP = ABP_MAX - ABP_MIN;
    ABP = (ABP - ABP_MIN)/ABP_AMP;
    
    if c == 1
        save(train_path_mat,'PPG','VPG','APG','ECG','ABP','SBP','DBP','ABP_GRND','ABP_RNorm','ABP_AMP','ABP_MIN','-v7.3');
        %
        h5create(train_path_hdf5,'/PPG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ABP',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/VPG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/APG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ECG',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/SBP',[1 length(PPG)]);
        h5create(train_path_hdf5,'/DBP',[1 length(PPG)]);
        h5create(train_path_hdf5,'/ABP_AMP',[1 1]);
        h5create(train_path_hdf5,'/ABP_MIN',[1 1]);
        h5create(train_path_hdf5,'/ABP_GRND',[output_signal_length length(PPG)]);
        h5create(train_path_hdf5,'/ABP_RNorm',[output_signal_length length(PPG)]);
        h5write(train_path_hdf5,'/PPG',PPG);
        h5write(train_path_hdf5,'/ABP',ABP);
        h5write(train_path_hdf5,'/VPG',VPG);
        h5write(train_path_hdf5,'/APG',APG);
        h5write(train_path_hdf5,'/ECG',ECG);
        h5write(train_path_hdf5,'/SBP',SBP);
        h5write(train_path_hdf5,'/DBP',DBP);
        h5write(train_path_hdf5,'/ABP',ABP);
        h5write(train_path_hdf5,'/ABP_AMP',ABP_AMP);
        h5write(train_path_hdf5,'/ABP_MIN',ABP_MIN);
        h5write(train_path_hdf5,'/ABP_GRND',ABP_GRND);
        h5write(train_path_hdf5,'/ABP_RNorm',ABP_RNorm);
    elseif c == 2
        save(test_path_mat,'PPG','VPG','APG','ECG','ABP','SBP','DBP','ABP_GRND','ABP_AMP','ABP_MIN','-v7.3');
        %
        h5create(test_path_hdf5,'/PPG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ABP',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/VPG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/APG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ECG',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/SBP',[1 length(PPG)]);
        h5create(test_path_hdf5,'/DBP',[1 length(PPG)]);
        h5create(test_path_hdf5,'/ABP_AMP',[1 1]);
        h5create(test_path_hdf5,'/ABP_MIN',[1 1]);
        h5create(test_path_hdf5,'/ABP_GRND',[output_signal_length length(PPG)]);
        h5create(test_path_hdf5,'/ABP_RNorm',[output_signal_length length(PPG)]);
        h5write(test_path_hdf5,'/PPG',PPG);
        h5write(test_path_hdf5,'/ABP',ABP);
        h5write(test_path_hdf5,'/VPG',VPG);
        h5write(test_path_hdf5,'/APG',APG);
        h5write(test_path_hdf5,'/ECG',ECG);
        h5write(test_path_hdf5,'/SBP',SBP);
        h5write(test_path_hdf5,'/DBP',DBP);
        h5write(test_path_hdf5,'/ABP_AMP',ABP_AMP);
        h5write(test_path_hdf5,'/ABP_MIN',ABP_MIN);
        h5write(test_path_hdf5,'/ABP_GRND',ABP_GRND);
        h5write(test_path_hdf5,'/ABP_RNorm',ABP_RNorm);
    end
end