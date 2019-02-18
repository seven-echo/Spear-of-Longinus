function [U_updated, U_S_updated,U_A_updated] = spear_update(Net, Net_d, Attri, Attri_d, U_list, U_A_list, beta1, beta2, alpha)
%SPEAR_UPDATE ���ѵõ��ĵ�άǶ����и���
%   �˴���ʾ��ϸ˵��
%          Net   is the weighted adjacency matrix
%         Attri  is the attribute information matrix with row denotes nodes
%         beta   is the weight for node structure information
%         alpha  is the weights for different order structure information

%% update structure embeddings
U_list_updated = RandNE_Update(Net, Net_d, U_list);
U_S_updated = RandNE_Combine(U_list_updated, alpha);

%% update attribute embeddings
U_A_updated = RandNE_Update(Attri, Attri_d, U_A_list);

%% updated embeddings
U_updated = beta1*(U_S_updated)+beta2*(U_A_updated);

end

