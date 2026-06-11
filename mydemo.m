%% ================= 初始化 =================
clear;
clc;
warning off all;
addpath(genpath('./'));

load WebKB.mat

nCluster = length(unique(Y));
n_view = length(X);

for i = 1:n_view
    X{i} = X{i}';
end


k = 2;
n = size(X{1},2);

%% ================= 结果表 =================
ResultTable = table( ...
    [], [], [], [], [], ...
    [], [], ...
    [], [], ...
    [], [], ...
    [], [], ...
    [], [], ...
    'VariableNames',{ ...
    'alpha','gamma','self_q','cross_q','m', ...
    'ACC_mean','ACC_std', ...
    'NMI_mean','NMI_std', ...
    'Fscore_mean','Fscore_std', ...
    'Purity_mean','Purity_std', ...
    'Time_mean','Time_std' ...
    });

%% ================= 最优记录 =================
ACCMAX = 0;

best_alpha = 0;
best_gamma = 0;
best_self_q = 0;
best_cross_q = 0;
best_m = 0;

best_nmi = 0;
best_fscore = 0;
best_purity = 0;
best_time = 0;

best_acc_std = 0;
best_nmi_std = 0;
best_fscore_std = 0;
best_purity_std = 0;
best_time_std = 0;

%% ================= 参数搜索 =================
p1 = [0.0001 0.001 0.01 0.1 1 10 100 1000];

for alpha = p1

    for gamma = [0.0001 0.001 0.01 0.1 1 10 100 ]

        for self_q = [0.0001 0.001 0.01 0.1 1 10 100 1000]

            for cross_q = [0.0001 0.001 0.01 0.1 1 10 100 1000]

                p5 = [2*nCluster 3*nCluster 4*nCluster 5*nCluster];

                for m = p5

                    %% ================= 参数设置 =================
                    [W, BB, centers] = ConstructBipartiteGraph(X,m,k);
                    para.alpha = alpha;
                    para.gamma = gamma;
                    para.self_q = self_q;
                    para.cross_q = cross_q;

                    para.c = nCluster;
                    para.d = 4*nCluster;
                    para.m = m;

                    %% ================= 运行20次 =================
                    num_run = 20;

                    ACC_all = zeros(num_run,1);
                    NMI_all = zeros(num_run,1);
                    Fscore_all = zeros(num_run,1);
                    Purity_all = zeros(num_run,1);
                    Time_all = zeros(num_run,1);

                    for run_idx = 1:num_run

                        rng(run_idx);

                        tic;
                        [yy1, iter_num, obj] = SMCOBG(X,W,para,nCluster);
                        Time_all(run_idx) = toc;

                        result = ClusteringMeasure1(Y,yy1);

                        ACC_all(run_idx) = result(1);
                        NMI_all(run_idx) = result(2);
                        Fscore_all(run_idx) = result(4);
                        Purity_all(run_idx) = result(5);

                    end

                    %% ================= Mean ± Std =================
                    ACC_mean = mean(ACC_all);
                    ACC_std  = std(ACC_all);

                    NMI_mean = mean(NMI_all);
                    NMI_std  = std(NMI_all);

                    Fscore_mean = mean(Fscore_all);
                    Fscore_std  = std(Fscore_all);

                    Purity_mean = mean(Purity_all);
                    Purity_std  = std(Purity_all);

                    Time_mean = mean(Time_all);
                    Time_std  = std(Time_all);

                    %% ================= 保存结果 =================
                    newRow = { ...
                        alpha, gamma, self_q, cross_q, m, ...
                        ACC_mean, ACC_std, ...
                        NMI_mean, NMI_std, ...
                        Fscore_mean, Fscore_std, ...
                        Purity_mean, Purity_std, ...
                        Time_mean, Time_std ...
                        };

                    ResultTable = [ResultTable; newRow];

                    writetable(ResultTable,'result.xlsx');

                    %% ================= 更新最优结果 =================
                    if ACC_mean > ACCMAX

                        ACCMAX = ACC_mean;

                        best_alpha = alpha;
                        best_gamma = gamma;
                        best_self_q = self_q;
                        best_cross_q = cross_q;
                        best_m = m;

                        best_nmi = NMI_mean;
                        best_fscore = Fscore_mean;
                        best_purity = Purity_mean;
                        best_time = Time_mean;

                        best_acc_std = ACC_std;
                        best_nmi_std = NMI_std;
                        best_fscore_std = Fscore_std;
                        best_purity_std = Purity_std;
                        best_time_std = Time_std;

                    end

                    %% ================= 当前结果输出 =================
                    fprintf('\n====================================================\n');

                    fprintf('alpha: %.4f | gamma: %.4f | self_q: %.4f | cross_q: %.4f | m: %d\n', ...
                        alpha,gamma,self_q,cross_q,m);

                    fprintf('----------------------------------------------------\n');

                    fprintf('ACC:     %.2f ± %.4f\n', ...
                        ACC_mean*100, ACC_std);

                    fprintf('NMI:     %.2f ± %.4f\n', ...
                        NMI_mean*100, NMI_std);

                    fprintf('Fscore:  %.2f ± %.4f\n', ...
                        Fscore_mean*100, Fscore_std);

                    fprintf('Purity:  %.2f ± %.4f\n', ...
                        Purity_mean*100, Purity_std);

                    fprintf('Time:    %.2f ± %.2f seconds\n', ...
                        Time_mean, Time_std);

                    fprintf('Current Best ACC: %.2f\n', ...
                        ACCMAX*100);

                    fprintf('====================================================\n');

                end
            end
        end
    end
end

%% ================= 最终结果 =================
fprintf('\n==================== FINAL RESULT ====================\n');

fprintf('Best ACC: %.2f\n\n', ACCMAX*100);

fprintf('Best Params:\n');
fprintf('alpha: %.4f\n', best_alpha);
fprintf('gamma: %.4f\n', best_gamma);
fprintf('self_q: %.4f\n', best_self_q);
fprintf('cross_q: %.4f\n', best_cross_q);
fprintf('m: %d\n', best_m);

fprintf('\nMean ± Std Statistics:\n');

fprintf('ACC:    %.2f ± %.4f\n', ...
    ACCMAX*100, best_acc_std);

fprintf('NMI:    %.2f ± %.4f\n', ...
    best_nmi*100, best_nmi_std);

fprintf('Fscore: %.2f ± %.4f\n', ...
    best_fscore*100, best_fscore_std);

fprintf('Purity: %.2f ± %.4f\n', ...
    best_purity*100, best_purity_std);

fprintf('Time:   %.2f ± %.2f\n', ...
    best_time, best_time_std);

fprintf('====================================================\n');
