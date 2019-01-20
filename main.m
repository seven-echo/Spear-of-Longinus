clear;
Dataset = 'Flickr';
fprintf('Running on %s\n', Dataset);
% 自动化调参测试代码 
% 先只遍历alpha参数，参数共5个值，测试其从0到100，间隔为4，共有25的5次方
% for a = (0:9:50)
%     for b = (0:9:50)
%         for c = (0:9:50)
%             for d1 = (0:9:50)
%                 e = 1;
%                     if(e==1&&d1==0&&c==0&&b==0&&a==0)
%                         % 强行设置alpha初始值，接着之前的进程继续遍历寻参
%                         e = 1;
%                         d1 = 9;
%                         c = 9;
%                         b = 9;
%                         a = 9;
%                     end
%                     alpha = [e,d1,c,b,a];
%                     %开始遍历
for i = (1:3)
t_start = tic;
if strcmp(Dataset,'BlogCatalog')
    load('BlogCatalog.mat')
    q = 5; % the order for structure similarity
    beta1 = 0.1; % weight of network structure information for embedding
    beta2 = 8; % weight of network attribute information for embedding
    Ortho = 1; % ortho is whether orthogonalization the projection matrix
    alpha = [10,100,0.1,1,0.2]; % alpha is the weights for different order structure information
    seed = 0; % seed is the seed of random for generate the projection matrix
    %delta1 = 2; % weight of network information for constructing test representation H2
    %delta2 = 1; % weight of node attribute information for constructing test representation H2
    topk = 50; % topk of node attribute matrix
elseif strcmp(Dataset,'Flickr')
    load('Flickr.mat')
    q = 4; % the order for structure similarity
    beta1 = 0.1; % weight of network structure information for embedding
    beta2 = 0.51; % weight of network attribute information for embedding
    Ortho = 1; % ortho  is whether orthogonalization the projection matrix
    alpha = [0,1,10,1,1.2]; % alpha is the weights for different order structure information
    
    
    seed = 0; % seed is the seed of random for generate the projection matrix
%     delta1 = 2.1; % weight of network information for constructing test representation H2
%     delta2 = 1; % weight of node attribute information for constructing test representation H2
    topk = 100; % topk of node attribute matrix
end

d = 512; % the dimension of the embedding representation
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
%% Training group
G1 = sparse(W(Group1,Group1)); % network of nodes in the training group
A1 = sparse(A(Group1,:)); % node attributes of nodes in the training group
Y1 = sparse(Y(Group1,:)); % labels of nodes in the training group
%% Test group
A2 = sparse(A(Group2,:)); % node attributes of nodes in the test group
GC1 = sparse(W(Group1,:)); % For constructing test representation H2
GC2 = sparse(W(Group2,:)); % For constructing test representation H2
G2 = sparse(W(Group2,Group2));

%% Unsupervised Attributed Network Embedding (LANE w/o Label)
disp('Unsupervised Attributed Network Embedding (Random Projection):')

[H, end_time, U_list, U_S] = spear(W,A,d,beta1,beta2,alpha,Ortho,seed,q,topk);
H1 = H(Group1,:);
H2 = H(Group2,:);

% [H1, U_S, U_A] = spear(G1, A1, d, beta1, beta2, alpha, Ortho, seed, q); % representation of training group
% H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1)); % representation of test group
t_embi(i) = toc(t_start);
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
data = [beta1, beta2, topk, alpha, F1macro, F1micro, rem(end_time,60), rem(t_end, 60)];
dlmwrite('arguments_test.dat',data,'-append');

% %遍历结束
%             end
%         end
%     end
% end
% fprintf('time:%d minutes and %f seconds\n',floor(end_time/60),rem(end_time,60));
% fprintf('embedding time is %d minutes and %f seconds\n',floor(t_emb/60),rem(t_emb,60));
% fprintf('total time is %d minutes and %f seconds\n',floor(t_end/60),rem(t_end,60));
