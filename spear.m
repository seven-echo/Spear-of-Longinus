function [U, end_time,U_list,U_S] = spear(Net, Attri, d, beta1, beta2, alpha, Ortho, seed, q, topk)

% 通过随机投影来进行网络的表示
%          Net   is the weighted adjacency matrix
%         Attri  is the attribute information matrix with row denotes nodes
%        LabelY  is the label information matrix
%          d     is the dimension of the embedding representation
%         beta   is the weight for node structure information
%         alpha  is the weights for different order structure information
%         ortho  is whether orthogonalization the projection matrix
%         seed   is the seed of random for generate the projection matrix
%          q     is the order of structure similarity

%% Random Projection
start_time = tic;
N = length(Net);

% calculate structure embedding matrix U_S
W_tran = Net;
U_list = RandNE_Projection(W_tran,q,d,Ortho,seed);
U_S = RandNE_Combine(U_list,alpha);
R = cell2mat(U_list(1));
% 

% calculate attribute similarity matrix U_A -------------------
A_T = Attri';
A_T = bsxfun(@rdivide, A_T, sum(A_T.^2).^.5); % Normalize
S_A = A_T'*A_T; % attritute cosine similarity matrix
sparse_S_A = zeros(N,N);
% calculate sparse attribute matrix
for i = (1:N)
    [value,~] = sort(S_A(i,:),'descend');
    [~,Yaxis] = find(S_A(i,:)>=value(topk));
    sparse_S_A(i,Yaxis) = S_A(i,Yaxis);
end

A_tran = sparse_S_A;
U_A = A_tran * R;
% -----------------------U_A-----------------------------------

% Obtain mixture embedding
U = beta1*(U_S)+beta2*(U_A);
end_time = toc(start_time);
fprintf('Random Projection is end. time:%d minutes and %f seconds\n',floor(end_time/60),rem(end_time,60));
% Random Projection end