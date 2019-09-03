%% USING SVM AND MLP CLASSIFIERS TO PREDICT HADRON OBSERVATIONS IN RADIO EMISSION DATA
% Using MAGIC dataset found at the UCI repository: https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.names
% Stephanie Jury

clc, clear all, close all
rng(1) % Set random seed

%% --- 1. IMPORT DATA ---

filename = '/Users/mac/Desktop/NeuralComputing/Coursework9/MAGIC_data.csv';
delimiter = ',';

% Read columns of data as text:
formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';
% Open the text file.
fileID = fopen(filename,'r','n','UTF-8');
% Skip the BOM (Byte Order Mark).
fseek(fileID, 3, 'bof');
% Read columns of data according to the format.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
% Close the text file.
fclose(fileID);

% Convert the contents of columns containing numeric text to numbers.
% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));
for col=[1,2,3,4,5,6,7,8,9,10,11]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^[-/+]*\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end

% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells
MAGIC = table;

% Define variables
MAGIC.fLength = cell2mat(raw(:, 1)); % major axis of ellipse [mm]
MAGIC.fWidth = cell2mat(raw(:, 2)); % minor axis of ellipse [mm] 
MAGIC.fSize = cell2mat(raw(:, 3)); % 10-log of sum of content of all pixels [in #phot]
MAGIC.fConc = cell2mat(raw(:, 4)); % ratio of sum of two highest pixels over fSize  [ratio]
MAGIC.fConc1 = cell2mat(raw(:, 5)); % ratio of highest pixel over fSize  [ratio]
MAGIC.fAsym = cell2mat(raw(:, 6)); % distance from highest pixel to center, projected onto major axis [mm]
MAGIC.fM3Long = cell2mat(raw(:, 7)); % 3rd root of third moment along major axis  [mm] 
MAGIC.fM3Trans = cell2mat(raw(:, 8)); % 3rd root of third moment along minor axis  [mm]
MAGIC.fAlpha = cell2mat(raw(:, 9)); % angle of major axis with vector to origin [deg]
MAGIC.fDist = cell2mat(raw(:, 10)); % distance from origin to center of ellipse [mm]

% Define class
MAGIC.class = cell2mat(raw(:, 11)); % gamma (signal), hadron (background)

% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp R;

%% --- 2. INSPECT DATA AND PREPROCCESS ---

% Inspect table properties
MAGIC_array = table2array(MAGIC);

% Inspect MAGIC dataset variable properties
MAGIC_mean = mean(MAGIC_array); 
MAGIC_std = std(MAGIC_array); MAGIC_skewness = skewness(MAGIC_array); 
MAGIC_min = min(MAGIC_array); MAGIC_max = round(max(MAGIC_array),8);

header_inspect = {'Mean','Standard Deviation', 'Skewness','Minimum', 'Maximum'};
MAGIC_summary = [MAGIC_mean' MAGIC_std' MAGIC_skewness' MAGIC_min' MAGIC_max'];
MAGIC_summary_table = [header_inspect; num2cell(MAGIC_summary)]

% Inspect MAGIC dataset variable properties by class (gamma)
MAGIC_array_1 = MAGIC_array(MAGIC_array(:,11) == 1, :);
MAGIC_mean_1 = mean(MAGIC_array_1); MAGIC_std_1 = std(MAGIC_array_1);
MAGIC_skewness_1 = skewness(MAGIC_array_1); MAGIC_min_1 = min(MAGIC_array_1);
MAGIC_max_1 = round(max(MAGIC_array_1),8);
MAGIC_summary_1 = [MAGIC_mean_1' MAGIC_std_1' MAGIC_skewness_1' MAGIC_min_1' MAGIC_max_1'];
MAGIC_summary_table_1 = [header_inspect; num2cell(MAGIC_summary_1)]

% Inspect MAGIC dataset variable properties by class (noise)
MAGIC_array_0 = MAGIC_array(MAGIC_array(:,11) == 0, :);
MAGIC_mean_0 = mean(MAGIC_array_0); MAGIC_std_0 = std(MAGIC_array_0);
MAGIC_skewness_0 = skewness(MAGIC_array_0); MAGIC_min_0 = min(MAGIC_array_0);
MAGIC_max_0 = round(max(MAGIC_array_0),8);
MAGIC_summary_0 = [MAGIC_mean_0' MAGIC_std_0' MAGIC_skewness_0' MAGIC_min_0' MAGIC_max_0'];
MAGIC_summary_table_0 = [header_inspect; num2cell(MAGIC_summary_0)]

% Inspect size of the array
MAGIC_size = size(MAGIC_array)

% Check for missing entries in variable data
idx_t = MAGIC_array(:,1:10)==NaN;
MAGIC_missing_values =sum(idx_t(:))

% Inspect variable properties (raw)
figure(1)
set(gcf, 'Position',  [200, 200, 1000, 800], 'color','w');
hist_list = [MAGIC.fLength, MAGIC.fWidth, MAGIC.fSize, MAGIC.fConc, MAGIC.fConc1,...
    MAGIC.fAsym, MAGIC.fM3Long, MAGIC.fM3Trans, MAGIC.fAlpha, MAGIC.fDist];
title_list = ['Ellipse Major Axis', 'Ellipse Minor Axis', 'Log(10)sum(Pixel Content)',...
    {'sum(Two Highest Pixels)','/fSize'}, {'Highest Pixel','/fSize'}, {'Highest Pixel to Center','(Major Axis)'},...
    {'Third Root of Third Moment','(Major Axis)'},{'Third Root of','Third Moment (Minor Axis)'}, ...
    {'Angle of Major Axis','with Vector to Origin'},'Distance from Origin to Center' ];
for subplot_number = 1:10
  subplot(2,5,subplot_number);
  hist(hist_list(:, subplot_number),50);
  title(title_list(subplot_number));
end

hold off

% Inspect variable properties (normalised)
MAGIC{:, 1:10} = normalize(MAGIC{:, 1:10});

figure(2)
set(gcf, 'Position',  [200, 200, 1000, 800], 'color','w');
hist_list = [MAGIC.fLength, MAGIC.fWidth, MAGIC.fSize, MAGIC.fConc, MAGIC.fConc1,...
    MAGIC.fAsym, MAGIC.fM3Long, MAGIC.fM3Trans, MAGIC.fAlpha, MAGIC.fDist]; 
title_list = ['Ellipse Major Axis', 'Ellipse Minor Axis', 'Log(10)sum(Pixel Content)',...
    {'sum(Two Highest Pixels)','/fSize'}, {'Highest Pixel','/fSize'}, {'Highest Pixel to Center','(Major Axis)'},...
    {'Third Root of Third Moment','(Major Axis)'},{'Third Root of','Third Moment (Minor Axis)'}, ...
    {'Angle of Major Axis','with Vector to Origin'},'Distance from Origin to Center' ];
for subplot_number = 1:10
  subplot(2,5,subplot_number);
  hist(hist_list(:, subplot_number),50);
  title(title_list(subplot_number));
end

hold off

%% Split data into pulsar and noise categories and inspect distribution by class

gamma_data = MAGIC(MAGIC.class == 1, :);
noise_data = MAGIC(MAGIC.class == 0, :);

hist_list_gamma = [gamma_data.fLength, gamma_data.fWidth, gamma_data.fSize, gamma_data.fConc, gamma_data.fConc1,...
    gamma_data.fAsym, gamma_data.fM3Long, gamma_data.fM3Trans, gamma_data.fAlpha, gamma_data.fDist];

hist_list_noise = [noise_data.fLength, noise_data.fWidth, noise_data.fSize, noise_data.fConc, noise_data.fConc1,...
    noise_data.fAsym, noise_data.fM3Long, noise_data.fM3Trans, noise_data.fAlpha, noise_data.fDist];

title_list_short = [{'fLength'}, {'fWidth'}, {'fSize'},{'fConc'},{'fConc1'},{'fAsym'},{'fM3Long'},{'fM3Trans'},...
    {'fAlpha'},{'fDist'}];

figure(3)
set(gcf, 'Position',  [200, 200, 1000, 800], 'color','w');

for subplot_number = 1:10
  subplot(2,5,subplot_number);
  plot((hist(hist_list_gamma(:, subplot_number),50)), 'b');
  hold on
  plot((hist(hist_list_noise(:, subplot_number),50)), 'r');
  title(title_list_short(subplot_number));
end

hold off

%% Inspect the degree of bias towards the majority class (gamma)

gamma_class = sum(MAGIC.class(:) == 1);
[row, col] = size(MAGIC.class);
gamma_bias = round((gamma_class/row * 100),2) % Shows the data is biased towards gamma (signal) data

%% Correlation plot and heatmap

% figure(4)
% corr = corrplot(HTRU2)
% title('Dataset Feature and Class Correlation Plot')
% hold off

figure(5)
set(gcf,'color','w','position',[73,147,807,602] );
MAGIC_array = table2array(MAGIC);
corr_matrix = corrcoef(MAGIC_array);
imagesc(corr_matrix); 
set(gca, 'XTick'); 
set(gca, 'YTick');
xticklabels({'fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha',...
    'fDist','Class'});
xtickangle(305);
yticklabels({'fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha',...
    'fDist','Class'});
set(gca, 'YTickLabel');
title('Feature and Class Correlation Matrix');
colormap('hot'); 
colorbar;
hold off

%% Split a hold-out set and training/validation set

[m,n] = size(MAGIC);
p = 0.70; % Puts 70% of the whole dataset into training/validation set and 30% into a hold-out set for testing
idx_t = randperm(m);
training = MAGIC(idx_t(1:round(p*m)),:); 
testing = MAGIC(idx_t(round(p*m)+1:end),:);

% Convert data to array type
training = table2array(training); 
testing = table2array(testing);
% save('test_set.mat', 'testing');
%% Remove signal data to make training data less biased towards this class

gamma_count = sum(training(:,11) == 1); % Sum of gamma classifications in training set
gamma_class = training((training(:,11) == 1),:); % Gamma data only
noise_count = sum(training(:,11) == 0); % Sum of noise classifications in training set
noise_class = training((training(:,11) == 0),:); % Noise data only

% Create a new training set which is evenly distributed between gamma and noise data
training_unbiased_ordered = [gamma_class(1:noise_count,:); noise_class]; 

% Shuffle ordered training data so it can be sliced for cross-validation
training_unbiased = training_unbiased_ordered(randperm(size(training_unbiased_ordered,1)),:);

% Check data loss and remaining training data
data_loss = size((training),1) - size((training_unbiased),1)
size((training_unbiased),1)

%% --- 3. CREATE k-FOLD CROSS VALIDATION SETS ---

k = 3; % Set number of folds
[rows, cols] = size(training_unbiased);

test_slice = round(rows/(k+1)); % Testing data slice size depends on k and length of training data 
% e.g. a testing slice size 200 from a dataset of 2000 rows

training_slice = rows - test_slice; % Training data size is the remainder after test slice removed 
% e.g. remainder is 1800 rows

% Define the first slice
% This is always a special case with test data on top and training data underneath
val_test_1 = training_unbiased(1:test_slice,:);
val_train_1 = training_unbiased(test_slice+1:end,:);

% Initialise variables to store middle slices
val_test = [val_test_1];
val_train = [val_train_1];

for i = 2:k
    % Set test data slice: e.g. take test data from position 201 to position 400 
    new_val_test = training_unbiased(test_slice*(i-1)+1:test_slice*i,:);
    
    % Training data remains above and below the test slice
    % Take training data from position 1 to 200 and concatenate with data
    % from position 401 to the end of the training data
    new_val_train = [training_unbiased(1:test_slice*(i-1),:); training_unbiased(test_slice*i+1:end,:)];
    
    % Add the new cross validation sets to val_test and val_train
    val_test = [val_test ; new_val_test];
    val_train = [val_train ; new_val_train];
end

% Define the final (kth) slice
% This is always a special case with test data on the bottom and training on top
val_test_k = training_unbiased((k-1)*test_slice+1:end,:);
val_train_k = training_unbiased(1:(k-1)*test_slice,:);

% Create vectors which contain all k test and training sets
% Loop through these when applying models
val_test = [val_test ; val_test_k];
val_train = [val_train ; val_train_k];

%% --- 4. DETERMINE WHICH SVM MODEL PARAMETERS PERFORM BEST IN VALIDATION

data = table('Size',[0,15], 'VariableNames', {'Model', 'Box_Constraint', 'Kernel_Scale', 'Accuracy',...
    'Mean_Error', 'Runtime', 'Num_TN', 'Num_FN', 'Num_TP', 'Num_FP', 'TPR', 'TNR', 'PPV', 'NPV', 'F1' }, 'VariableTypes',...
    ["string", "double", "double","double" "double", "double", "double", "double", "double", "double",... 
    "double", "double", "double", "double", "double"]);

% Set models to test
models = [{'gaussian'}, {'poly2'}, {'poly3'}, {'poly4'}, {'linear'}];

% Set box constraint values to test 
box_constraint = [1];

% Set kernel scale values to test
kernel_scale = [1];

for m = 1:length(models)

    for s = 1:length(kernel_scale)

        for b = 1:length(box_constraint)

            model_accuracy_store = []; model_error_store = []; model_TN_store = []; model_FN_store = [];
            model_TP_store = []; model_FP_store = []; model_TPR_store = []; model_TNR_store = [];
            model_PPV_store = []; model_NPV_store = []; model_F1_store = []; model_runtime_store = [];

            strcat("Model: ", string(models(m)),"    Kernel Scale: ", string(kernel_scale(s)),...
                "    Box Constraint: ", string(box_constraint(b)))
            
            for i = 1:k % Train and test model on each k-fold

                % Set variables and classes in the validation training slice
                x = val_train(training_slice*i - training_slice + 1:training_slice*i,1:10);
                y = val_train(training_slice*i - training_slice + 1:training_slice*i,11);

                % Train SVM
                if string(models(m)) == 'gaussian' ||string(models(m)) == 'linear' 
                    tic % Record model training time
                    SVM_mdl = fitcsvm(x,y,'BoxConstraint', box_constraint(b), 'KernelFunction',...
                        string(models(m)), 'KernelScale', kernel_scale(s));
                    toc;
                elseif string(models(m)) == 'poly2'
                    tic % Record model training time
                    SVM_mdl = fitcsvm(x,y,'BoxConstraint', box_constraint(b), 'KernelFunction',...
                        'Polynomial','PolynomialOrder', 2, 'KernelScale', kernel_scale(s));
                    toc;
                elseif string(models(m)) == 'poly3'
                    tic % Record model training time
                    SVM_mdl = fitcsvm(x,y,'BoxConstraint', box_constraint(b), 'KernelFunction',...
                        'Polynomial','PolynomialOrder', 3, 'KernelScale', kernel_scale(s));
                    toc;
                elseif string(models(m)) == 'poly4'
                    tic % Record model training time
                    SVM_mdl = fitcsvm(x,y,'BoxConstraint', box_constraint(b), 'KernelFunction',...
                        'Polynomial','PolynomialOrder', 4, 'KernelScale', kernel_scale(s));
                    toc;
                else
                    print('No model selected')
                    exit
                end
                
                % Select validation set to pass to model for prediction
                val_slice = val_test(test_slice*i - test_slice + 1:test_slice*i,1:11);

                % Test model on the validation set and store posterior probabilities
                [SVM_predicted_class_gaussian, SVM_posterior_gaussian] = predict(SVM_mdl,val_slice(:,1:10));

                % Calculate model validation set error rates
                true_class = val_slice(:,11);
                correct_class = sum(true_class == SVM_predicted_class_gaussian);
                model_accuracy_calc = correct_class / length(true_class);
                model_error_calc = 1 - (correct_class / length(true_class));

                % Calculate model validation set error metrics
                [SVM_cv_gaussian,SVM_cv_order_gaussian] = confusionmat(true_class,SVM_predicted_class_gaussian);
                model_TN = SVM_cv_gaussian(1,1); model_FN = SVM_cv_gaussian(2,1);
                model_FP = SVM_cv_gaussian(1,2); model_TP = SVM_cv_gaussian(2,2);
                model_TPR = model_TP./(model_TP + model_FN);
                model_TNR = model_TN./(model_TN + model_FP);
                model_PPV = model_TP./(model_TP + model_FP);
                model_NPV = model_TN./(model_TN + model_FN);
                model_F1 = (2*model_TP)./(2*model_TP + model_FP + model_FN);
                

                % Store validation set data
                model_accuracy_store = [model_accuracy_store; model_accuracy_calc];
                model_error_store = [model_error_store; model_error_calc];
                model_TN_store = [model_TN_store; model_TN]; model_FN_store = [model_FN_store; model_FN];
                model_TP_store = [model_TP_store; model_TP]; model_FP_store = [model_FP_store; model_FP];
                model_TPR_store = [model_TPR_store; model_TPR]; model_TNR_store = [model_TNR_store; model_TNR];
                model_PPV_store = [model_PPV_store; model_PPV]; model_NPV_store = [model_NPV_store; model_NPV];
                model_F1_store = [model_F1_store; model_F1]; model_runtime_store = [model_runtime_store; round(toc,2)];

            end

            model_accuracy =  mean(model_accuracy_store); model_error = mean(model_error_store);
            model_TN = mean(model_TN_store); model_FN = mean(model_FN_store);
            model_TP = mean(model_TP_store); model_FP = mean(model_FP_store);
            model_TPR = mean(model_TPR_store); model_TNR = mean(model_TNR_store);
            model_PPV = mean(model_PPV_store); model_NPV = mean(model_NPV_store);
            model_F1 = mean(model_F1_store); model_runtime = mean(model_runtime_store);

            output = table(models(m), box_constraint(b), kernel_scale(s), model_accuracy, model_error, model_runtime, model_TN, model_FN,...
                model_TP, model_FP, model_TPR, model_TNR, model_PPV, model_NPV, model_F1, ...
                'VariableNames', {'Model', 'Box_Constraint', 'Kernel_Scale', 'Accuracy',...
                'Mean_Error','Runtime', 'Num_TN', 'Num_FN', 'Num_TP', 'Num_FP', 'TPR', 'TNR', 'PPV', 'NPV', 'F1'});
            data = [data; output];

        end
    end
end

data

%% Colourmap plot of gris search results

data_raw = readtable('SVM_Grid_Search.csv'); % Requires loading results csv so as not to run all models
accuracy_threshold = 0.1; % Set minimum accuracy threshold to observe
data = data_raw(data_raw.Accuracy >= accuracy_threshold, :);
accuracy_max = data_raw(data_raw.Accuracy == max(data_raw.Accuracy), :);

models = ["gaussian", "linear", "poly2", "poly3","poly4"];
titles = ["Gaussian", "Linear", "Polynomial: 2nd Order", "Polynomial: 3rd Order","Polynomial: 4th Order"];

figure(6)

for subplot_number = 1:length(models)
    subplot(3,2,subplot_number);
    model_data = data((strcmp(data.Model, models(subplot_number))), 2:end);
    mesh_x = model_data.BoxConstraint;
    mesh_y = model_data.KernelScale;
    mesh_z = model_data.Accuracy.*100;
    set(gcf,'color','w');
    tri = delaunay(mesh_x,mesh_y);
    trisurf(tri,mesh_x,mesh_y,mesh_z); 
    view(2);
    hold on 
    shading interp
    caxis manual
    caxis([40 85]);
    colormap(hot)
    %colormap(flipud(parula))
    xlim([0 2]);
    ylim([0 2]);
    xlabel('Box Constraint');
    ylabel('Kernel Scale');
    zlabel('Mean Classification Accuracy %');
    if max(mesh_z) == accuracy_max.Accuracy*100
        idxmax = find(mesh_z == max(mesh_z));
        h = scatter3(mesh_x(idxmax),mesh_y(idxmax),mesh_z(idxmax),'b','filled');
        h.SizeData = 36;
        legend(h,{['Maximum Mean' newline 'Classification Accuracy']});
    end
    str = sprintf(titles(subplot_number));
    title(str)
end

%% --- 5. DETERMINE WHICH MLP MODEL PARAMETERS PERFORM BEST IN VALIDATION

% 5.1 Read data in the right format:
% Prepare input and targets 
end_train = size(training_unbiased,1);% Find the size of the training set for slicing
input = training_unbiased(:,1:10);% Get input and target in the right format:
target_pre = training_unbiased(:,11);
target = [target_pre == 0 target_pre]; % Turn targets into binary outputs

% Read inputs and targets in the right format:
x = input';
t = target';

% 5.2 Set parameters outside of grid-search:
processFcn = {'removeconstantrows','mapstd'};
divideFcn = 'divideind'; %Divide the training set according to pre-defined indices
performFcn = 'mse';% We use 'mse' as trainlm cannot use 'crossentropy'
init_state = 'default'; %Weight and bias initialization options: %'rand(0,1)' 'rand(-1,1)' 'normal' 'default'
transferFcn_outputlayer = 'softmax'; % Calculate probability for binary classes
transferFcn = 'tansig'; % 'tansig' 'logsig'
max_validation_fail = 50; %Early stopping condition: Max number of validations to fail to decrease
time_limit = inf; % Time limit (to control time whie tuning parameters)

% 5.3 Parameters tuning inside grid-search:
trainFcn_set = {'traingdm','traingdx','trainscg','trainlm'};% Choose a training function from: % 'traingdm', 'traingdx', 'trainlm', 'trainscg'
hidlaysize = [10 15 20]; % for all training functions
learning_rates = [0.03 0.01 0.03]; % traingdm and traingdx
momentum = [0.3 0.5 0.9]; %traingdm and traingdx
mu = [0.003 0.01 0.03]; % trainlm

% 5.4 Set up grid search:
% Loop through the grid
for b = 1:length(trainFcn_set) % Loop through each train function
% Prepare placeholders to store training results for each parameters combination(a):
% Set up empty matrices:
    MLP_validation_results = [];
    MLP_run_array = []; 
    neurons = [];
    learning_rate = [];
    momentum_rate = [];
    mu_rate = [];
    sigma_rate = [];
    MLP_cvs_class_error = [];
    MLP_cvs_TPR = [];
    MLP_cvs_TNR = [];
    MLP_cvs_PPV = [];
    MLP_cvs_NPV = [];
    MLP_cvs_F1 = [];
    MLP_sbest_epoch = [];
    MLP_sperformance_measure = [];
    MLP_cvs_runtime = [];
    MLP_perf_record = [];

% Set up the grid
    access_trainFcn = trainFcn_set(b);
    trainFcn = access_trainFcn{1};
    if strcmpi(trainFcn, 'trainlm') % trainlm parameters
        [p, s] = ndgrid(hidlaysize, mu); % number of neurons and mu
        pairs = [p(:) s(:)];
    elseif (strcmpi(trainFcn, 'traingdm')) || (strcmpi(trainFcn, 'traingdx')) % traingdm and traingdx parameters: 
        [p,q,r] = ndgrid(hidlaysize,learning_rates,momentum);% number of neurson, learning rates and momentum
        pairs = [p(:) q(:) r(:)];
    elseif strcmpi(trainFcn, 'trainscg')
        [p, u] = ndgrid(hidlaysize, sigma);
        pairs = [p(:) u(:)];
    end;
    fprintf('Length of grid is: %d \n', size(pairs,1))
% for each training function loop through the grid:
    for a = 1:size(pairs,1)
        MLP_run = a;
        % Set up the network according to the grid:
        % Choose number of hidden neurons for each layer:
        MLP_net = patternnet([pairs(a,1) pairs(a,1)], trainFcn, performFcn);
        % Choose learning rates & momentum if gdm & gdx
        % Choose mu if lm
        % Choose sigma if scg
        if strcmpi(trainFcn, 'traingdx')
          MLP_net.trainParam.lr = pairs(a,2);
          MLP_net.trainParam.mc = pairs(a,3);
        elseif strcmpi(trainFcn, 'traingdm')
          MLP_net.trainParam.lr = pairs(a,2);
          MLP_net.trainParam.mc = pairs(a,3);
        elseif strcmpi(trainFcn, 'trainlm')
          MLP_net.trainParam.mu = pairs(a,2);
        elseif strcmpi(trainFcn, 'trainscg')
          MLP_net.trainParam.sigma = pairs(a,2);  
        end;

        % Preprocess inputs and outputs:
        MLP_net.input.processFcns = processFcn;
        % Divide Training data into Training and Validation
        MLP_net.divideFcn = divideFcn;%'divideind'
        % Transfer function at output layer:    
        MLP_net.layers{3}.transferFcn = transferFcn_outputlayer;  
        % Activation functions: 'tansig' or 'logsig'
        MLP_net.layers{1}.transferFcn = transferFcn;
        MLP_net.layers{2}.transferFcn = transferFcn;

        % Weight & bias initialization:
        if strcmpi(init_state, 'rand(-1,1)') % Rand(-1,1)
          initFcn_input_weight = -1 + 2.*rand(pairs(a,1),size(x,1));
          initFcn_layer_weight_2 = -1 + 2.*rand(pairs(a,1),pairs(a,1));
          initFcn_layer_weight_3 = -1 + 2.*rand(size(t,1),pairs(a,1));
          initFcn_input_bias = -1 + 2.*rand(pairs(a,1),1);
          initFcn_layer_bias_2 = -1 + 2.*rand(pairs(a,1),1);
          initFcn_layer_bias_3 = -1 + 2.*rand(size(t,1),1);
        elseif strcmpi(init_state, 'rand(0,1)') %Rand(0,1)
          initFcn_input_weight = rand(pairs(a,1),size(x,1));
          initFcn_layer_weight_2 = rand(pairs(a,1),pairs(a,1));
          initFcn_layer_weight_3 = rand(size(t,1),pairs(a,1));
          initFcn_input_bias = rand(pairs(a,1),1);
          initFcn_layer_bias_2 = rand(pairs(a,1),1);
          initFcn_layer_bias_3 = rand(size(t,1),1);
        elseif strcmpi(init_state, 'normal') % Special Normal (See Neural Networks: Tricks of Trade, p.20)
          initFcn_input_weight = normrnd(0,size(x,1)^(-1/2),[pairs(a,1),size(x,1)]);
          initFcn_layer_weight_2 = normrnd(0,pairs(a,1)^(-1/2),[pairs(a,1),pairs(a,1)]);
          initFcn_layer_weight_3 = normrnd(0,pairs(a,1)^(-1/2),[size(t,1),pairs(a,1)]);
          initFcn_input_bias = normrnd(0,1^(-1/2),[pairs(a,1),1]);
          initFcn_layer_bias_2 = normrnd(0,1^(-1/2),[pairs(a,1),1]);
          initFcn_layer_bias_3 = normrnd(0,1^(-1/2),[size(t,1),1]);
        end;

        % Set the early stopping condition for max validation fails
        MLP_net.trainParam.max_fail = max_validation_fail;
        % Set the maximum time to train while tuning parameters:
        MLP_net.trainParam.time = time_limit;
        % List of all plot functions type:
        % Set the plots to draw for eacht training:
        MLP_net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
            'plotconfusion', 'plotroc'};

        % Store output for each loop (combination fo parameters)
        MLP_validation_testslice = [];
        MLP_cv_class_error = [];
        MLP_cv_TPR = [];
        MLP_cv_TNR = [];
        MLP_cv_PPV = [];
        MLP_cv_NPV = [];
        MLP_cv_F1 = [];
        MLP_cv_runtime = [];
        MLP_best_epoch = [];
        MLP_performance_measure = [];
        MLP_loop_perf_record = [];

        % k-fold cross-validation for each loop:
        for i = 1:k % k=10
            % Set up 
            if i == 1
                MLP_net.divideParam.trainInd = test_slice+1:end_train;
                MLP_net.divideParam.valInd = 1:test_slice;
            elseif i == k
                MLP_net.divideParam.trainInd = 1:(i-1)*test_slice;
                MLP_net.divideParam.valInd = (i-1)*test_slice+1:end_train;
            else
                MLP_net.divideParam.trainInd = [1:test_slice*(i-1) test_slice*i+1:end_train];
                MLP_net.divideParam.valInd = test_slice*(i-1)+1:test_slice*i;
            end;


            % Select validation test data to pass to model for prediction
            val_slice = val_test(test_slice*i - test_slice + 1:test_slice*i,1:11);

            % Store test slice set number
            MLP_validation_testslice = [MLP_validation_testslice; i];

            % Weight and Biases re-initialization for each iteration:
            % Default option is the Nguyen-Widrow init function:
            if strcmpi(init_state, 'default')
                MLP_net = init(MLP_net);
            else
                % configure the network to set initialize weights and biases
                % to a pre-defined distributed range
                MLP_net = configure(MLP_net, x, t);
                MLP_net.iw{1,1} = initFcn_input_weight;
                MLP_net.b{1} = initFcn_input_bias;
                MLP_net.lw{2,1} = initFcn_layer_weight_2;
                MLP_net.b{2} = initFcn_layer_bias_2;
                MLP_net.lw{3,2} = initFcn_layer_weight_3;
                MLP_net.b{3} = initFcn_layer_bias_3;
            end;

             % Time how long the model takes to run
            tic

            % Train the network
            [MLP_net,tr] = train(MLP_net,x,t);
            % To store the best models:
            if tr.best_vperf < 0.11
                filename = strcat('MLP_',trainFcn,int2str(a),int2str(i),int2str(n));
                save(filename, 'MLP_net');
                fprintf(' Model %s has the performance error of: %d \n', filename, tr.best_vperf); 
            end;

            % Collect results for the kth fold: 
            time_record_k = tr.time;
            MLP_cv_runtime = [MLP_cv_runtime; time_record_k(end)];

            time_record_k = tr.time;
                MLP_cv_runtime = [MLP_cv_runtime; time_record_k(end)];

            % Mean Time-evolution:
            % Store the performance record for each fold:
            perf_record_k = tr.perf;

            % Collect the performance record of mse and the running time record
            % Following a predefined time line:
            time_line = (0:0.5:30);
            perf_record_tim_k = zeros(size(time_line,2),1)';
            for tim = 1:length(time_line)
                if tim == 1 
                    if isempty(find(time_record_k<=time_line(tim)))
                        perf_record_tim_k(tim) = 0;
                    else
                        index_perf = find(time_record_k<=time_line(tim));
                        perf_record_tim_k(tim) = mean(perf_record_k(index_perf));
                    end;
                else
                    if isempty(find(...
                        time_record_k<=time_line(tim) & time_record_k>time_line(tim-1)))
                        perf_record_tim_k(tim) = 0;
                    else
                        index_perf = find(...
                        time_record_k<=time_line(tim) & time_record_k >time_line(tim-1));    
                        perf_record_tim_k(tim) = mean(perf_record_k(index_perf));
                    end
                end;
             end;
             MLP_loop_perf_record = [MLP_loop_perf_record; perf_record_tim_k];


            % Test model on the validation test slice
            MLP_predicted_class = vec2ind(MLP_net(val_slice(:,1:10)'));    

            % Store validation set error rates
            init_true_class = val_slice(:,11);
            pre_true_class = [init_true_class == 0 init_true_class];
            true_class = vec2ind(pre_true_class');
            correct = sum(true_class == MLP_predicted_class);
            MLP_cv_class_error_new = 1 - (correct / length(true_class));
            MLP_cv_class_error = [MLP_cv_class_error ; MLP_cv_class_error_new];

            % Calculate other error metrics
            [MLP_cv_c,MLP_cv_order] = confusionmat(true_class',MLP_predicted_class');
            MLP_cv_TN = MLP_cv_c(1,1);
            MLP_cv_FN = MLP_cv_c(2,1);
            MLP_cv_FP = MLP_cv_c(1,2);
            MLP_cv_TP = MLP_cv_c(2,2);

            % Best epoch and lowest performance error:
            MLP_best_epoch = [MLP_best_epoch; tr.best_epoch];
            MLP_performance_measure = [MLP_performance_measure; tr.best_vperf];    

            % Calculate new precision/recall
            MLP_cv_TPR_new = MLP_cv_TP./(MLP_cv_TP+MLP_cv_FN);
            MLP_cv_TNR_new = MLP_cv_TN./(MLP_cv_TN+MLP_cv_FP);
            MLP_cv_PPV_new = MLP_cv_TP./(MLP_cv_TP+MLP_cv_FP);
            MLP_cv_NPV_new = MLP_cv_TN./(MLP_cv_TN+MLP_cv_FN);
            MLP_cv_F1_new = (2*MLP_cv_TP)./(2*MLP_cv_TP + MLP_cv_FP + MLP_cv_FN);

            % Store error metrics
            MLP_cv_TPR = [MLP_cv_TPR; MLP_cv_TPR_new];
            MLP_cv_TNR = [MLP_cv_TNR; MLP_cv_TNR_new];
            MLP_cv_PPV = [MLP_cv_PPV; MLP_cv_PPV_new];
            MLP_cv_NPV = [MLP_cv_NPV; MLP_cv_NPV_new];
            MLP_cv_F1 = [MLP_cv_F1; MLP_cv_F1_new];
        end;%>> End a kth fold

        % Collect results for each loop (a): (parameters combination)
        % Store the parameters for each loop
        MLP_run_array = [MLP_run_array; MLP_run];
        neurons = [neurons; pairs(a,1)];
        if strcmpi(trainFcn, 'trainlm')
            mu_rate = [mu_rate; pairs(a,2)];
        elseif (strcmpi(trainFcn, 'traingdm')) || (strcmpi(trainFcn, 'traingdx'))
            learning_rate = [learning_rate; pairs(a,2)];
            momentum_rate = [momentum_rate; pairs(a,3)];
        elseif strcmpi(trainFcn, 'trainscg')
            sigma_rate = [sigma_rate; pairs(a,2)];
        end;

        % Store the error and performance metrics for each loop
        MLP_cvs_class_error = [MLP_cvs_class_error; mean(MLP_cv_class_error)];
        MLP_cvs_TPR = [MLP_cvs_TPR; mean(MLP_cv_TPR)];
        MLP_cvs_TNR = [MLP_cvs_TNR; mean(MLP_cv_TNR)];
        MLP_cvs_PPV = [MLP_cvs_PPV; mean(MLP_cv_PPV)];
        MLP_cvs_NPV = [MLP_cvs_NPV; mean(MLP_cv_NPV)];
        MLP_cvs_F1 = [MLP_cvs_F1; mean(MLP_cv_F1)];
        MLP_sbest_epoch = [MLP_sbest_epoch; mean(MLP_best_epoch)];
        MLP_sperformance_measure = [MLP_sperformance_measure; mean(MLP_performance_measure)];
        MLP_cvs_runtime = [MLP_cvs_runtime; mean(MLP_cv_runtime)];
    fprintf('Finish loop number: %d \n', a)   
    end;
    
% 5.5 Collect results for all the loops:
    % Choose the best model by classification error rate:
    [MLP_best_error MLP_lowest_error_index] = min(MLP_cvs_class_error);
    fprintf('Best error run time %d:', MLP_lowest_error_index)

    % Store the results for each grid search
      Structure_results = struct('Results',MLP_table, 'Performance',...
      table(MLP_lowest_error_index, MLP_best_error,'VariableNames',{'Best_error_run_time','Min_Class_Error'}));

    % Store performance record for each function
    % per layer:
    filename_perf_record = strcat(trainFcn,'_perf_record');
    trainFcn_cell = cell(length(neurons),1);
    trainFcn_cell(:) = {trainFcn};
    if strcmpi(trainFcn, 'trainscg')
        learning_rate_plot = zeros(length(sigma_rate),1);
        momentum_rate_plot = zeros(length(sigma_rate),1);
        mu_rate_plot = zeros(length(sigma_rate),1);
        MLP_table_trainscg = table(trainFcn_cell, MLP_run_array,neurons,learning_rate_plot,...
            momentum_rate_plot,mu_rate_plot,sigma_rate,...
            MLP_cvs_class_error,MLP_cvs_TPR,...
            MLP_cvs_TNR,MLP_cvs_PPV,MLP_cvs_NPV,MLP_cvs_F1,...
            MLP_sbest_epoch, MLP_sperformance_measure,MLP_cvs_runtime,...
            MLP_perf_record,...
            'VariableNames',{'trainFcn_cell', 'MLP_run_array','neurons','learning_rate',...
            'momentum_rate','mu_rate','sigma_rate','MLP_cvs_class_error','MLP_cvs_TPR',...
            'MLP_cvs_TNR','MLP_cvs_PPV','MLP_cvs_NPV','MLP_cvs_F1',...
            'MLP_sbest_epoch', 'MLP_sperformance_measure','MLP_cvs_runtime',...
            'MLP_perf_record'});
        save(filename_perf_record,'MLP_table_trainscg');
    elseif strcmpi(trainFcn, 'trainlm')
        learning_rate_plot = zeros(length(mu_rate),1);
        momentum_rate_plot = zeros(length(mu_rate),1);
        sigma_rate_plot = zeros(length(mu_rate),1);
        MLP_table_trainlm = table(trainFcn_cell, MLP_run_array,neurons,learning_rate_plot,...
            momentum_rate_plot,mu_rate,sigma_rate_plot,MLP_cvs_class_error,MLP_cvs_TPR,...
            MLP_cvs_TNR,MLP_cvs_PPV,MLP_cvs_NPV,MLP_cvs_F1,...
            MLP_sbest_epoch, MLP_sperformance_measure,MLP_cvs_runtime,...
            MLP_perf_record,...
            'VariableNames',{'trainFcn_cell', 'MLP_run_array','neurons','learning_rate',...
            'momentum_rate','mu_rate','sigma_rate','MLP_cvs_class_error','MLP_cvs_TPR',...
            'MLP_cvs_TNR','MLP_cvs_PPV','MLP_cvs_NPV','MLP_cvs_F1',...
            'MLP_sbest_epoch', 'MLP_sperformance_measure','MLP_cvs_runtime',...
            'MLP_perf_record'});
        save(filename_perf_record,'MLP_table_trainlm');
    elseif strcmpi(trainFcn, 'traingdm')
        %momentum_rate_plot = zeros(length(learning_rate),1);
        mu_rate_plot = zeros(length(learning_rate),1);
        sigma_rate_plot = zeros(length(learning_rate),1);
        MLP_table_traingdm = table(trainFcn_cell,MLP_run_array, neurons,learning_rate,...
            momentum_rate,mu_rate_plot,sigma_rate_plot,MLP_cvs_class_error,MLP_cvs_TPR,...
            MLP_cvs_TNR,MLP_cvs_PPV,MLP_cvs_NPV,MLP_cvs_F1,...
            MLP_sbest_epoch, MLP_sperformance_measure,MLP_cvs_runtime,...
            MLP_perf_record,...
            'VariableNames',{'trainFcn_cell', 'MLP_run_array','neurons','learning_rate',...
            'momentum_rate','mu_rate','sigma_rate','MLP_cvs_class_error','MLP_cvs_TPR',...
            'MLP_cvs_TNR','MLP_cvs_PPV','MLP_cvs_NPV','MLP_cvs_F1',...
            'MLP_sbest_epoch', 'MLP_sperformance_measure','MLP_cvs_runtime',...
            'MLP_perf_record'});
        save(filename_perf_record,'MLP_table_traingdm');
    else
        %momentum_rate_plot = zeros(length(learning_rate),1);
        mu_rate_plot = zeros(length(learning_rate),1);
        sigma_rate_plot = zeros(length(learning_rate),1);
        MLP_table_traingdx = table(trainFcn_cell, MLP_run_array,neurons,learning_rate,...
            momentum_rate,mu_rate_plot,sigma_rate_plot,MLP_cvs_class_error,MLP_cvs_TPR,...
            MLP_cvs_TNR,MLP_cvs_PPV,MLP_cvs_NPV,MLP_cvs_F1,...
            MLP_sbest_epoch, MLP_sperformance_measure,MLP_cvs_runtime,...
            MLP_perf_record,...
            'VariableNames',{'trainFcn_cell', 'MLP_run_array','neurons','learning_rate',...
            'momentum_rate','mu_rate','sigma_rate','MLP_cvs_class_error','MLP_cvs_TPR',...
            'MLP_cvs_TNR','MLP_cvs_PPV','MLP_cvs_NPV','MLP_cvs_F1',...
            'MLP_sbest_epoch', 'MLP_sperformance_measure','MLP_cvs_runtime',...
            'MLP_perf_record'});
        save(filename_perf_record,'MLP_table_traingdx');
    end;
    
    % Save file name:
    filename_results = strcat(trainFcn,'_withoutPCA_',transferFcn,'_',init_state,...
        '_',performFcn,'_results.mat');
    filename_variables = strcat(trainFcn,'_withoutPCA_',transferFcn,'_',init_state,...
        '_',performFcn,'_variables.mat');
    save(filename_results, 'Structure_results');
    save(filename_variables);
    fprintf('Finish running %s \n', trainFcn)
end; % End the running time for the training function

% Full results table:
MLP_combine_table = vertcat(MLP_table_traingdm,MLP_table_traingdx,...
    MLP_table_trainscg,MLP_table_trainlm); 
% Save the table:
% save('MLP_combine_table','MLP_combine_table');
% writetable(MLP_combine_table,'MLP_Grid_Search.csv');

% Plot MSE vs. Time for each hidden layer size:
ii = 3;
hidlaysize = [10 15 20];
time_line = (0:0.5:30);
%     pre_matrix = MLP_combine_table(...
%         ((MLP_combine_table.neurons==hidlaysize(ii) & (strcmp(MLP_combine_table.trainFcn_cell,'trainscg')))|...
%         (MLP_combine_table.neurons==hidlaysize(ii) & (strcmp(MLP_combine_table.trainFcn_cell,'trainlm')))), :);
pre_matrix = MLP_combine_table(...
    (MLP_combine_table.neurons==hidlaysize(ii)),:);
matrix = pre_matrix.MLP_perf_record;
matrix(matrix==0) = NaN;
p = plot(time_line, matrix','linewidth',1.5);
set(p, {'color'}, num2cell(jet(size(matrix',2)),2));
hold on
trainFcn_col = pre_matrix.trainFcn_cell;
sigma_rate_col = pre_matrix.sigma_rate;
learning_rate_col = pre_matrix.learning_rate;
momen_rate_col = pre_matrix.momentum_rate;
mu_rate_col = pre_matrix.mu_rate;

legendCell={};
legendCell = {'traingdm(0.003,0.3)','traingdm(0.01,0.3)',...
    'traingdm(0.03,0.3)','traingdm(0.003,0.5)','traingdm(0.01,0.5)',...
    'traingdm(0.03,0.5)','traingdm(0.003,0.9)','traingdm(0.01,0.9)',...
    'traingdm(0.03,0.9)','traingdx(0.003,0.3)','traingdx(0.01,0.3)',...
    'traingdx(0.03,0.3)','traingdx(0.003,0.5)','traingdx(0.01,0.5)',...
    'traingdx(0.03,0.5)','traingdx(0.003,0.9)','traingdx(0.01,0.9)',...
    'traingdx(0.03,0.9)','trainscg(e-13)','trainscg(e-5)','trainscg(e-4)',...
    'trainlm(0.003)','trainlm(0.01)','trainlm(0.03)'};
legend_title = legend(legendCell);
set(legend_title,'NumColumns',2,'FontSize',15);
title_name = strcat('Performance Plots for Hidden Layers with ',...
    int2str(hidlaysize(ii)),' neurons');
title(title_name,'FontSize',30);
ylabel('Mean Square Error','FontSize',18);
xlabel({'Estimated Run Time per Model  in Seconds';...
    '(Maximum 1000 epochs) & (Maximum 50 failed validation checks)'},...
    'FontSize',15);
hold off;    
%% --- 6. APPLY TUNED SVM AND MLP MODELS TO HELD-OUT TEST DATA ---

% SVM
    % Train best model on all training data
    %SVM_test = fitcsvm(training(:,1:10),training(:,11),'BoxConstraint', 0.4, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'KernelScale',0.8);
    %save SVM_test SVM_test
    load SVM_test
    % Test model on the held out test data
    [SVM_test_predicted_class, SVM_test_posterior] = predict(SVM_test,testing(:,1:10));
    %SVM_test_predicted = (1 - SVM_test_posterior);
    SVM_test_plotconfusion = 1-SVM_test_posterior(:,2);
    
    % Store test set error rates
    test_true_class = testing(:,11);
    SVM_test_correct = sum(test_true_class == SVM_test_predicted_class);
    SVM_test_class_error = 1 - (SVM_test_correct / length(test_true_class));

    % Store true positive, true negative, precision, accuracy and F1 score
    [SVM_test_c,SVM_test_order] = confusionmat(test_true_class,SVM_test_predicted_class);
    
    % Inspect error details averaged over validation sets
    SVM_test_TN = SVM_test_c(1,1); SVM_test_FN = SVM_test_c(2,1);
    SVM_test_FP = SVM_test_c(1,2); SVM_test_TP = SVM_test_c(2,2);
    SVM_test_TPR = SVM_test_TP./(SVM_test_TP+SVM_test_FN);
    SVM_test_TNR = SVM_test_TN./(SVM_test_TN+SVM_test_FP);
    SVM_test_PPV = SVM_test_TP./(SVM_test_TP+SVM_test_FP);
    SVM_test_NPV = SVM_test_TN./(SVM_test_TN+SVM_test_FN);
    SVM_test_F1 = (2*SVM_test_TP)./(2*SVM_test_TP + SVM_test_FP + SVM_test_FN);

    header = {'Number TN', 'Number FN', 'Number TP','Number FP','TPR','TNR', 'PPV', 'NPV','F1','Accuracy', 'Mean Class. Error %'};
    SVM_test_error_details = [SVM_test_TN, SVM_test_FN, SVM_test_TP, SVM_test_FP, SVM_test_TPR,...
        SVM_test_TNR, SVM_test_PPV, SVM_test_NPV, SVM_test_F1];
    SVM_test_error_details_mean = [SVM_test_error_details,1-mean(SVM_test_class_error), mean(SVM_test_class_error)*100];
    SVM_test_error_details_table = [header; num2cell(SVM_test_error_details_mean)]
    
% MLP
    % Load the best model
    load MLP_test
    MLP_test = MLP_net;
   
    % Test model on the held out test data
    MLP_original_test_predicted = MLP_test(testing(:,1:10)'); 
    MLP_test_predicted_class = vec2ind(MLP_original_test_predicted);
    
    % Store test set error rates
    init_true_testclass = testing(:,11);
    pre_true_testclass = [init_true_testclass == 0 init_true_testclass];
    test_true_class = vec2ind(pre_true_testclass');
    MLP_test_correct = sum(test_true_class == MLP_test_predicted_class);
    MLP_test_class_error = 1 - (MLP_test_correct / length(test_true_class));

    % Store true positive, true negative, precision, accuracy and F1 score
    [MLP_test_c,MLP_test_order] = confusionmat(test_true_class,MLP_test_predicted_class);
    
    % Inspect error details averaged over validation sets
    MLP_test_TN = MLP_test_c(1,1); MLP_test_FN = MLP_test_c(2,1);
    MLP_test_FP = MLP_test_c(1,2); MLP_test_TP = MLP_test_c(2,2);
    MLP_test_TPR = MLP_test_TP./(MLP_test_TP+MLP_test_FN);
    MLP_test_TNR = MLP_test_TN./(MLP_test_TN+MLP_test_FP);
    MLP_test_PPV = MLP_test_TP./(MLP_test_TP+MLP_test_FP);
    MLP_test_NPV = MLP_test_TN./(MLP_test_TN+MLP_test_FN);
    MLP_test_F1 = (2*MLP_test_TP)./(2*MLP_test_TP + MLP_test_FP + MLP_test_FN);

    header = {'Number TN', 'Number FN', 'Number TP','Number FP','TPR','TNR', 'PPV', 'NPV','F1','Accuracy', 'Mean Class. Error %'};
    MLP_test_error_details = [MLP_test_TN, MLP_test_FN, MLP_test_TP, MLP_test_FP, MLP_test_TPR,...
        MLP_test_TNR, MLP_test_PPV, MLP_test_NPV, MLP_test_F1];
    MLP_test_error_details_mean = [MLP_test_error_details,1-mean(MLP_test_class_error), mean(MLP_test_class_error)*100];
    MLP_test_error_details_table = [header; num2cell(MLP_test_error_details_mean)]
      
%% --- 7. COMPARE BEST SVM AND MLP MODEL TEST SET PERFORMANCE ---

SVM_test_mean_stats = SVM_test_error_details_mean(5:10)
MLP_test_mean_stats = MLP_test_error_details_mean(5:10)

test_bar = [SVM_test_mean_stats(1) MLP_test_mean_stats(1);...
    SVM_test_mean_stats(2) MLP_test_mean_stats(2);...
    SVM_test_mean_stats(3) MLP_test_mean_stats(3);...
    SVM_test_mean_stats(4) MLP_test_mean_stats(4);...
    SVM_test_mean_stats(5) MLP_test_mean_stats(5);...
    SVM_test_mean_stats(6) MLP_test_mean_stats(6)];

x_label = {'TPR','TNR', 'PPV', 'NPV','F1','Accuracy'};
y_label = {'70%','75%','80%','85%','90%','95%','100%'};
y_ticks = linspace(0.7,1,7);

figure(9)
set(gcf,'color','w');
val_bar_chart = bar(test_bar);
set(gca,'XTickLabel',x_label,'YTick',y_ticks,'YTickLabel',y_label);
ylim([0.7 1]);
title('Best Model Accuracy in Testing');
legend('SVM', 'MLP');
hold off
 
%% Plot confusion matrices in testing
   
test_true_class_plotconfusion = [test_true_class-1 == 0; test_true_class-1]';

% SVM
    figure(10)
    SVM_test_posterior_plotconfusion = SVM_test_posterior';
    SVM_test_conf_plot = plotconfusion(test_true_class_plotconfusion', SVM_test_posterior_plotconfusion, 'SVM In Testing')
    fh = gcf;
    ah = fh.Children(2);
    ah.XTickLabel{1} = 'Noise';
    ah.XTickLabel{2} = 'Gamma';
    ah.YTickLabel{1} = 'Noise';
    ah.YTickLabel{2} = 'Gamma';
    ah.XLabel.String = 'Actual';
    ah.YLabel.String = 'Predicted';
    set(gcf,'color','w');
    hold off
    
% MLP
    figure(11)
    %MLP_test_scores_plotconfusion = MLP_test_scores';
    %MLP_test_conf_plot = plotconfusion(test_true_class_plotconfusion, MLP_test_scores_plotconfusion, 'MLP In Testing')
    MLP_test_conf_plot = plotconfusion(pre_true_testclass', MLP_original_test_predicted, 'MLP In Testing')
    fh = gcf;
    ah = fh.Children(2);
    ah.XTickLabel{1} = 'Noise';
    ah.XTickLabel{2} = 'Gamma';
    ah.YTickLabel{1} = 'Noise';
    ah.YTickLabel{2} = 'Gamma';
    ah.XLabel.String = 'Actual';
    ah.YLabel.String = 'Predicted';
    set(gcf,'color','w');
    hold off

%%  Plot ROC curves for tuned SVM and MLP model perfomance on held out test data
    
    % SVM
    [SVM_X,SVM_Y,SVM_T,SVM_AUC] = perfcurve(test_true_class,SVM_test_plotconfusion,1);
    SVM_AUC

    % MLP
    [MLP_X,MLP_Y,MLP_T,MLP_AUC] = perfcurve(test_true_class,MLP_original_test_predicted(2,:),2);
    MLP_AUC
    
    % Reference Line
    x_ROC = linspace(0,1,100);
    y_ROC = x_ROC;

    figure(12)
    set(gcf,'color','w');
    plot(x_ROC,y_ROC, '--k', 'HandleVisibility','off')
    hold on
    plot(SVM_X,SVM_Y)
    plot(MLP_X,MLP_Y)
    legend('SVM','MLP','Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for SVM and MLP in Testing')
    hold off
    