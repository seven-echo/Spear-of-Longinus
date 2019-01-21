clear;
Dataset = 'Flickr';
fprintf('Running on %s\n', Dataset);
% 自动化调参测试代码 
% 先只遍历alpha参数，参数共5个值，测试其从0到100，间隔为4，共有25的5次方
% for a = (0:0.001:0.1)
%     for b = (0:0.01:0.1)
%         for c = (0:0.1:1)
%             for d1 = (0:0.1:1)
% %                 e = 1;
% %                     if(e==1&&d1==0&&c==0&&b==0&&a==0)
% %                         % 强行设置alpha初始值，接着之前的进程继续遍历寻参
% %                         e = 1;
% %                         d1 = 9;
% %                         c = 9;
% %                         b = 9;
% %                         a = 9;
% %                     end
%                     alpha = [1,d1,c,b,a];
%                     %开始遍历

for i = (1:3)
t_start = tic;
if strcmp(Dataset,'BlogCatalog')
    load('BlogCatalog.mat')
    q = 4; % the order for structure similarity
    beta1 = 0.1; % weight of network structure information for embedding
    beta2 = 8; % weight of network attribute information for embedding
    Ortho = 1; % ortho is whether orthogonalization the projection matrix
    alpha = [1,1,0.05,0.05,0.001]; % alpha is the weights for different order structure information
%     alpha = [1,1,0.03,0.05,0.001]; % is good for full S_A matrix 
%     beta1 = 0.1 beta2 = 8;
    seed = 0; % seed is the seed of random for generate the projection matrix
    topk = 1000; % topk of node attribute matrix
elseif strcmp(Dataset,'Flickr')
    load('Flickr.mat')
    q = 4; % the order for structure similarity
    beta1 = 0.1; % 0.1 weight of network structure information for embedding
    beta2 = 70; % 0.51 weight of network attribute information for embedding
    Ortho = 1; % ortho  is whether orthogonalization the projection matrix
    alpha = [0.1,1,1,0.1,0.0000001]; % alpha is the weights for different order structure information
    seed = 0; % seed is the seed of random for generate the projection matrix
    topk = 1000; % topk of node attribute matrix
end

d = 100; % the dimension of the embedding representation
W = Network;
A = Attributes;
clear Attributes & Network

[n,~] = size(W); % Total number of nodes
W(1:n+1:n^2) = 1;
Y = [];
LabelIdx = unique(Label); % Indexes of all label categories
for n_Label_i = 1:length(LabelIdx)
    Y = [Y,Label==LabelIdx(n_Label_i)];
end
Y=Y*1;

Indices = randi(20,n,1); % 5-fold cross-validation indices
Group1 = find(Indices <= 16); % 1 for 1/16, 2 for 1/8, 4 for 1/4, 16 for 100% of training group
Group2 = find(Indices >= 17); % test group, test each fold in turns

%% Unsupervised Attributed Network Embedding (LANE w/o Label)
disp('Unsupervised Attributed Network Embedding (Random Projection):')
start_emb = tic;
[H, U_list, U_S, U_A, S_A, R] = spear(W,A,d,beta1,beta2,alpha,Ortho,seed,q,topk);
t_embi(i) = toc(start_emb);
H1 = H(Group1,:);
H2 = H(Group2,:);


[F1macro2,F1micro2] = Performance(H1,H2,Label(Group1,:),Label(Group2,:)); %
t_endi(i) = toc(t_start);
F1(i)=F1macro2;
F2(i)=F1micro2;
i = i+1;
end
t_emb = mean(t_embi);
F1macro = mean(F1)
F1micro = mean(F2)
t_end = mean(t_endi);
columns = {'beta1','beta2','topk','alpha','F1macro','F1micro','emb_time','totaltime'};
data = [beta1, beta2, topk, alpha, F1macro, F1micro, rem(t_emb,60), rem(t_emb, 60)];
dlmwrite('arguments_test_1_21.dat',data,'-append');

% %遍历结束
%             end
%         end
%     end
% end

% fprintf('embedding time is %d minutes and %f seconds\n',floor(t_emb/60),rem(t_emb,60));
% fprintf('total time is %d minutes and %f seconds\n',floor(t_end/60),rem(t_end,60));
