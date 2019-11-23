

clear;
clc;
clf;
currentFolder = pwd;
path = "/Users/zouyao/Documents/NEUfiles/MachineLearning/Exam2/Q1.csv";
data = csvread(path);
figure(1)
gscatter(data(:,1),data(:,2),data(:,3),'rk','ox');
legend('class -1', 'class 1')
xlabel('measurement x1')
ylabel('measurement x2')
grid on
%hold on
%%%%hold out for the first 10% data
TestData = data(1:100,:);
TrainData = data(101:end,:);
% mdlTree = fitctree(TrainData(:,1:2),TrainData(:,3),'MaxNumSplits',11,'SplitCriterion','gdi');

% % 
%  view(mdlTree,'mode','graph')
% % 
% predictLabel = predict(mdlTree,TestData(:,1:2));
% figure(2)
% C = confusionmat(TestData(:,3),predictLabel);
% confusionchart(C);

%ClassTreeEns = fitensemble(TrainData(:,1:2),TrainData(:,3),'AdaBoostM1',7,'Tree');

tree = calculID3(TrainData,0,7);
depth = calculateDepth(tree);
global boundary;
boundary = ones(2,depth+1);
traversTree(tree);
plotTree(tree,data,boundary);

for i = 1:100
    labelPre(i) = DecisionTreePredict(tree,TestData(i,1:2));
end

figure
D = confusionmat(TestData(:,3),labelPre);
confusionchart(D);  


%%%%bagging
for i = 1:7
    bagerror=0;
    resample(:,i) = randsample(900,900,true);
    trainbaggingdata(:,:,i) = TrainData(resample(:,i),:);
    baggingtree(i) = calculID3(trainbaggingdata(:,:,i),0,7);
    
    for j = 1:100
    labelPrebagging(j,i) = DecisionTreePredict(baggingtree(i),TestData(j,1:2));
        if labelPrebagging(j,i)~=TestData(j,3)
        end
    end
end
for i =1:100
    find1 = find(labelPrebagging(i,:) == 1);
    find2 = find(labelPrebagging(i,:) == -1);
    if size(find1,2) > size(find2,2)
        labelPrebagging(i,8) = 1;
    else
        labelPrebagging(i,8) = -1;
    end
end
figure
B = confusionmat(TestData(:,3),labelPrebagging(:,8));
confusionchart(B);

figure;
index = 1;
[x,y] = meshgrid(-1.5:0.01:1.5,-1.5:0.01:1.5); %grid of x and y indices
xy = [x(:) y(:)]; %x-y pairs
for i = 1:size(xy,1)
    for j = 1:7
        labelPrebagging(i,j) = DecisionTreePredict(baggingtree(j),xy(i,:));
    end
    find1 = find(labelPrebagging(i,:) == 1);
    find2 = find(labelPrebagging(i,:) == -1);
    if size(find1,2) > size(find2,2)
        labelPrebagging(i,8) = 1;
    else
        labelPrebagging(i,8) = -1;
    end
end
L = reshape(labelPrebagging(:,8),size(x));
%contourf(x, y, L,1)
hold on
[row,col] = size(L);
for i =2:row
    for j = 2:col
        if L(i,j) == 1 
            if L(i-1,j)+L(i-1,j-1)+L(i,j-1)+L(i+1,j)+L(i+1,j+1)+L(i,j+1)+L(i-1,j+1)+L(i+1,j-1) ==-2
               point(index,:) = [x(1,j),y(i,1),j,i,L(i,j)-L(i,j+1),L(i,j)-L(i,j-1),L(i,j)-L(i-1,j),L(i,j)-L(i+1,j)];
               index = index+1;
            end
            if L(i-1,j)+L(i-1,j-1)+L(i,j-1)+L(i+1,j)+L(i+1,j+1)+L(i,j+1)+L(i-1,j+1)+L(i+1,j-1) == 6
                point(index,:) = [x(1,j),y(i,1),j,i,L(i,j)-L(i,j+1),L(i,j)-L(i,j-1),L(i,j)-L(i-1,j),L(i,j)-L(i+1,j)];
                index = index+1;
            end
        end
    end
end
gscatter(data(:,1),data(:,2),data(:,3),'rk','ox');
xlabel('measurement x1')
ylabel('measurement x2')
checkindex = 1;
for i = 1:size(point,1)
    plotpoint(i,:) = [point(checkindex,1),point(checkindex,2)];
    point(checkindex,1) = 100;
    point(checkindex,2)= 100;
    checkdistance = (plotpoint(i,1) - point(:,1)).^2 +(plotpoint(i,2) - point(:,2)).^2;
    [~,minindex] = min(checkdistance);
    checkindex = minindex;
end
plotpoint(size(point,1)+1,:) = plotpoint(1,:);
plot(plotpoint(:,1),plotpoint(:,2),'LineWidth',4);
legend('class -1','class 1','bagging decision boundray')

%Z = labelxy.reshape(x.shape);


 
w = [0.1,0.2,0.4,0.2,0.1];
level = 7;
a = ones(1,level);
amdl = ones(1,level);
error = zeros(1,level);
errormdl = zeros(1,level);
weight = ones(1,900).*(1/900);
for i =1:level
    if i == 1
    abdaboostsampledata(:,:,i) = TrainData(:,:);
    adaboostTree(i) = calculID3(abdaboostsampledata(:,:,i),0,7);
    adaboostmdl = fitctree(abdaboostsampledata(:,1:2,i),abdaboostsampledata(:,3,i),'MaxNumSplits',11,'SplitCriterion','gdi');
    mdlpredict(:,i) = predict(adaboostmdl,abdaboostsampledata(:,1:2));
    for j = 1:900
        predictample(j) = DecisionTreePredict(adaboostTree(i),abdaboostsampledata(j,1:2,i));
        
        if  predictample(j) ~= abdaboostsampledata(j,3,i)
            error(i) = error(i)+weight(j);
        end
        
        if mdlpredict(j,i) ~= abdaboostsampledata(j,3,i)
            errormdl(i) = errormdl(i)+weight(j);
        end
    end
    a(i) = log((1-error(i))/error(i));
    amdl(i) = log((1-errormdl(i))/errormdl(i));
    for j =1:900
        if predictample(j) ~= abdaboostsampledata(j,3,i)
            weight(j) = weight(j)*exp(a(i));
        end
    end
    
    %%%normalize weight
    weight= weight./sum(weight);
    else
    %[bootstat,bootsam]= bootstrp(1,@mean,TrainData,'Weights',weight);
    bootsam = randsample(900,900,true,weight);
    abdaboostsampledata(:,:,i) = TrainData(bootsam,:);
    adaboostTree(i) = calculID3(abdaboostsampledata(:,:,i),0,7);
    adaboostmdl = fitctree(abdaboostsampledata(:,1:2,i),abdaboostsampledata(:,3,i),'MaxNumSplits',11,'SplitCriterion','gdi');
    mdlpredict(:,i) = predict(adaboostmdl,abdaboostsampledata(:,1:2));
    for j = 1:900
        predictample(j) = DecisionTreePredict(adaboostTree(i),abdaboostsampledata(j,1:2,i));
        
        if  predictample(j) ~= abdaboostsampledata(j,3,i)
            error(i) = error(i)+weight(j);
        end
        
        if mdlpredict(j,i) ~= abdaboostsampledata(j,3,i)
            errormdl(i) = errormdl(i)+weight(j);
        end
    end
    a(i) = log((1-error(i))/error(i));
    amdl(i) = log((1-errormdl(i))/errormdl(i));
    for j =1:900
        if predictample(j) ~= abdaboostsampledata(j,3,i)
            weight(j) = weight(j)*exp(a(i));
        end
    end
    
    %%%normalize weight
    weight= weight./sum(weight);
    end
end

uniqueweight = unique(weight);
figure
%color = [linspace(0,1,size(uniqueweight,2));linspace(0,1,size(uniqueweight,2));1-linspace(0,1,size(uniqueweight,2))]';
%color = [linspace(0,1,1/2*size(uniqueweight,2));linspace(0,1,size(uniqueweight,2));1-linspace(0,1,size(uniqueweight,2))]';
color = colormap;
incres = floor(64/size(uniqueweight,2));
clorbar = linspace(0,1,size(uniqueweight,2));
colorstr = cell(1,size(uniqueweight,2));
for i = 1:size(uniqueweight,2)
    str = sprintf('%4.4f',uniqueweight(i));
    colorstr(i) = cellstr(str);
    findweight = find(weight == uniqueweight(i));
    for j = 1:size(findweight,2)
        if TrainData(findweight(j),2) == 1
        scatter(TrainData(findweight(j),1),TrainData(findweight(j),2),30,color(i*incres,:));
        hold on
        else
            scatter(TrainData(findweight(j),1),TrainData(findweight(j),2),30,color(i*incres,:));
        hold on
        end
    end
end
xlabel('measurement x1')
ylabel('measurement x2')
title('updated weight')
% c = colorbar;
% scalor = 1/max(weight);
% colormatrix = [ones(1,900);weight.*scalor;zeros(1,900)]';
% figure 
% for i = 1: 900
% scatter(TrainData(i,1), TrainData(i,2), 1, colormatrix(i,:));
% hold on
% end

colorbar('Ticks',clorbar,'TickLabels',colorstr);

predictAdaboost = zeros(1,100);
for i = 1:100
    for j = 1:level
        predictAdaboost(i) = predictAdaboost(i)+ a(j)*DecisionTreePredict(adaboostTree(j),TestData(i,1:2));
    end
    if predictAdaboost(i)>0
        predictAdaboost(i) = 1;
    else
        predictAdaboost(i) = -1;
    end
end
figure
ada = confusionmat(TestData(:,3),predictAdaboost);
confusionchart(ada);

%%%%%plot the contoure of adaboost

figure;
index = 1;
[x,y] = meshgrid(-1.5:0.01:1.5,-1.5:0.01:1.5); %grid of x and y indices
xy = [x(:) y(:)]; %x-y pairs
labelPreadaboost = zeros(1,size(xy,1));
for i = 1:size(xy,1)
    for j = 1:7
        labelPreadaboost(i) = labelPreadaboost(i)+ a(j)*DecisionTreePredict(adaboostTree(j),xy(i,:));
    end
    if labelPreadaboost(i)>0
        labelPreadaboost(i) = 1;
    else
        labelPreadaboost(i) = -1;
    end
end

L = reshape(labelPreadaboost,size(x));
%contourf(x, y, L,1)
hold on
[row,col] = size(L);
for i =2:row
    for j = 2:col
        if L(i,j) == 1 
            if L(i-1,j)+L(i-1,j-1)+L(i,j-1)+L(i+1,j)+L(i+1,j+1)+L(i,j+1)+L(i-1,j+1)+L(i+1,j-1) ==-2
               point(index,:) = [x(1,j),y(i,1),j,i,L(i,j)-L(i,j+1),L(i,j)-L(i,j-1),L(i,j)-L(i-1,j),L(i,j)-L(i+1,j)];
               index = index+1;
            end
            if L(i-1,j)+L(i-1,j-1)+L(i,j-1)+L(i+1,j)+L(i+1,j+1)+L(i,j+1)+L(i-1,j+1)+L(i+1,j-1) == 6
                point(index,:) = [x(1,j),y(i,1),j,i,L(i,j)-L(i,j+1),L(i,j)-L(i,j-1),L(i,j)-L(i-1,j),L(i,j)-L(i+1,j)];
                index = index+1;
            end
        end
    end
end
gscatter(data(:,1),data(:,2),data(:,3),'rk','ox');
xlabel('measurement x1')
ylabel('measurement x2')
checkindex = 1;
for i = 1:size(point,1)
    plotpoint(i,:) = [point(checkindex,1),point(checkindex,2)];
    point(checkindex,1) = 100;
    point(checkindex,2)= 100;
    checkdistance = (plotpoint(i,1) - point(:,1)).^2 +(plotpoint(i,2) - point(:,2)).^2;
    [~,minindex] = min(checkdistance);
    checkindex = minindex;
end
plotpoint(size(point,1)+1,:) = plotpoint(1,:);
plot(plotpoint(:,1),plotpoint(:,2),'LineWidth',4);
legend('class -1','class 1','adaboost decision boundray')



function  traversTree(tree)
global boundary;
if tree.identity ~= 2
    boundary(1,boundary(1,1)+1) = tree.feature;
    boundary(2,boundary(1,1)+1) = tree.value;
    boundary(1,1) = boundary(1,1)+1;
    fprintf("smaller than x%d %f go to left else go to right \n",tree.feature,tree.value);
    traversTree(tree.left);
    traversTree(tree.right);
else
    fprintf("label is %d\n", tree.label);
end

end