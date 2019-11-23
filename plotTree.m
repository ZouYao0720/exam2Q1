function plotTree(tree,data,boundary)

%%%first calculate the depth


depth = calculateDepth(tree);


figure
hold on
axis([-4*depth 4*depth -2 4*depth+3]);
plotdecisiontree(tree,0,4*depth,depth);
title("the structure of decision tree",'FontSize',16)
set(gca,'xtick',[]);
set(gca,'ytick',[]);
hold off

figure

drawBound = boundary(:,2:boundary(1,1));
boundary1 = drawBound(:,find(drawBound(1,:)==1));
boundary2 = drawBound(:,find(drawBound(1,:)==2));
hold on
index = 1;
gscatter(data(:,1),data(:,2),data(:,3),'rk','ox');
xlabel('measurement x1')
ylabel('measurement x2')

x = [-0.98335,-0.002855,-0.002855,0.739606,0.739606,0.97499,0.97499,0.73906,0.73906,0.096393,0.096393,-0.98335,-0.98335];
y = [0.96288,0.96288,0.75469,0.75469,-0.057081,-0.057081,-0.83047,-0.83047,-0.96323,-0.96323,-0.83047,-0.83047,0.96288];

%pgon=polyshape(x,y);
plot(x,y,'LineWidth',4)
legend('class -1','class 1','decision boundary');
hold off
grid on
end

