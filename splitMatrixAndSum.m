function rowSums = splitMatrixAndSum(originalMatrix, k)
    % ��ȡԭ����Ĵ�С
    [N, M] = size(originalMatrix);
    
    % ��� M �Ƿ���Ա� k ����
    if mod(M, k) ~= 0
        error('���� M �����ܹ��� k ������');
    end
    
    % ����ÿ��С���������
    colsPerMatrix = M / k;
    
    % ��ʼ��һ���������洢ÿ��С��������͵Ľ��
    rowSums = zeros(N, k);
    
    % ���в��ԭ�������
    for i = 1:k
        % ���㵱ǰС�����������
        colStart = (i-1) * colsPerMatrix + 1;
        colEnd = i * colsPerMatrix;
        
        % ��ȡС����
        smallMatrix = originalMatrix(:, colStart:colEnd);
        
        % ��С���������
        rowSums(:, i) = sum(smallMatrix, 2); % �ڶ������� 2 ��ʾ�������
    end
end
