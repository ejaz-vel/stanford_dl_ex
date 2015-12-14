function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
poolSize = convolvedDim / poolDim;
pooledFeatures = zeros(poolSize, poolSize, numFilters, numImages);
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
for imageNum = 1:numImages
	for filterNum = 1:numFilters
    		pooledImage = zeros(poolSize, poolSize);
		for i = 1 : poolSize
			for j = 1 : poolSize
				x = ((i-1) * poolSize) + 1;
				y = ((j-1) * poolSize) + 1;
				pooledImage(i,j) = max(convolvedFeatures(x:x+poolSize-1, y:y+poolSize-1, filterNum, imageNum)(:));
			end
		end
		pooledFeatures(:, :, filterNum, imageNum) = pooledImage;
	end
end
end

