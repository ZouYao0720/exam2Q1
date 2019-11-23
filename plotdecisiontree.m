function plotdecisiontree(tree,x,y,depth)
if tree.identity == 1
   plot(x,y,'kd','MarkerSize',12);
   if tree.left.identity == 1 && tree.right.identity == 1
       x_left = x-depth*3; 
       y_left = y-4;
       x_right = x+depth*3;
       y_right = y-4;
   else
       x_left = x-depth; 
       y_left = y-4;
       x_right = x+depth;
       y_right = y-4;
   end
       
   quiver(x,y,x_left-x,y_left-y,'k')
   quiver(x,y,x_right-x,y_right-y,'k')
   plotdecisiontree(tree.left,x_left,y_left,depth-1);
   plotdecisiontree(tree.right,x_right,y_right,depth-1);
   ptext = sprintf("x%d < %f", tree.feature,tree.value);
   text(x-2,y+1,ptext,'FontSize',12);
else
    plot(x,y,'k.','MarkerSize',24);
    ptext = sprintf("%d",tree.label);
    text (x,y-0.5,ptext,'FontSize',12);
end
