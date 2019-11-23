function [tree] = calculID3(TrainData,depth,maxdepth)

%%%%%creat tree node

tree = struct('feature', 'null','value','null','identity','null', 'depth','null','left', 'null', 'right', 'null');

[m,n] = size(TrainData);
numfeature = n-1;
label = TrainData(:,n);
labelID = unique(label);

Bestentropy = calculateEntropy(TrainData);


%%%% select feature
featureEntropy = zeros(numfeature,2);
for i = 1:numfeature
    featureTemp = unique(TrainData(:,i));
    numF = length(featureTemp);%%%%calculate the num of samples under this feature
    newentropy = zeros(1,numF);
    for j = 1:numF
        find1 = find(TrainData(:,i)>=featureTemp(j));
        find2 = find(TrainData(:,i)<featureTemp(j));
        datasplit1 = TrainData(find1,:);
        datasplit2 = TrainData(find2,:);
        newentropy(j) =Bestentropy-( length(find1)/m*calculateEntropy(datasplit1) + length(find2)/m*calculateEntropy(datasplit2));
    end
    [maxEntropy,bestvalue] = max(newentropy);
    featureEntropy(i,:) =  [maxEntropy,featureTemp(bestvalue)];
end

[~,bestfeature] = max(featureEntropy(:,1));
tree.identity = 1;
tree.feature = bestfeature;
tree.value = featureEntropy(bestfeature,2);
tree.depth = depth+1;

left = TrainData(find(TrainData(:,tree.feature)<=tree.value),:);
right = TrainData(find(TrainData(:,tree.feature)>tree.value),:);
ginileft = calcuGini(left);
giniright = calcuGini(right);
numoflabeleft = length(unique(left(:,3)));

if tree.depth > maxdepth
    leaf = struct('label', 'null', 'identity','null');
    if sum(TrainData(:,3))>0 
            leaf.label = 1; 
    else 
            leaf.label = -1;
    end
    leaf.identity = 2;
    tree = leaf;
else
if isempty(left)
    leaf = struct('label', 'null', 'identity','null');
    if sum(right(:,3))>0 
            leaf.label = 1; 
    else 
            leaf.label = -1;
    end
    leaf.identity = 2;
    tree = leaf;
else
    if isempty(right)
        leaf = struct('label', 'null', 'identity','null');
        if sum(left(:,3))>0 
            leaf.label = 1; 
        else 
            leaf.label = -1;
        end
        leaf.identity = 2;
        tree = leaf;
    else
        tree.left = calculID3(left,tree.depth,maxdepth);
        tree.right = calculID3(right,tree.depth,maxdepth);
    end
end
end

%%% check for trees
if tree.identity == 1
if tree.left.identity == 2 && tree.right.identity == 2
    if tree.left.label == tree.right.label
       leaf = struct('label', 'null', 'identity','null');
       leaf.label = tree.left.label;
       leaf.identity = 2;
       tree = leaf;
    end
end
end
return
end