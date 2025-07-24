function dec_values = BIThightrans(bin_array)
    % ȷ��������������
    bin_array = bin_array(:)';
    
    % ������볤���Ƿ�Ϊ6�ı���
    if mod(length(bin_array), 6) ~= 0
        error('�������������ĳ��ȱ�����6�ı�����');
    end
    
    % ����������������Ϊ6��N�еľ���
    groups = reshape(bin_array, 6, []);
    
    % ����Ȩֵ������2^5 �� 2^0��
    weights = 2.^(5:-1:0);
    
    % ����ÿ���ʮ����ֵ
    dec_values = weights * groups;
    
    % ת��Ϊ���������
    dec_values = dec_values(:)';
end