function rowSums = splitMatrixAndSum(originalMatrix, k)
   
    [N, M] = size(originalMatrix);
    
   
    if mod(M, k) ~= 0
        error('列数 M 必须能够被 k 整除。');
    end
    
    
    colsPerMatrix = M / k;
    
    
    rowSums = zeros(N, k);
    
   
    for i = 1:k
        
        colStart = (i-1) * colsPerMatrix + 1;
        colEnd = i * colsPerMatrix;
        
        
        smallMatrix = originalMatrix(:, colStart:colEnd);
        
        
        rowSums(:, i) = sum(smallMatrix, 2); 
    end
end
