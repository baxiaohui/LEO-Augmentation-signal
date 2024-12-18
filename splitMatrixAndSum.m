function rowSums = splitMatrixAndSum(originalMatrix, k)
    % 获取原矩阵的大小
    [N, M] = size(originalMatrix);
    
    % 检查 M 是否可以被 k 整除
    if mod(M, k) ~= 0
        error('列数 M 必须能够被 k 整除。');
    end
    
    % 计算每个小矩阵的列数
    colsPerMatrix = M / k;
    
    % 初始化一个数组来存储每个小矩阵按行求和的结果
    rowSums = zeros(N, k);
    
    % 按列拆分原矩阵并求和
    for i = 1:k
        % 计算当前小矩阵的列索引
        colStart = (i-1) * colsPerMatrix + 1;
        colEnd = i * colsPerMatrix;
        
        % 提取小矩阵
        smallMatrix = originalMatrix(:, colStart:colEnd);
        
        % 对小矩阵按行求和
        rowSums(:, i) = sum(smallMatrix, 2); % 第二个参数 2 表示按行求和
    end
end
