function depth = calculateDepth(tree)
    if tree.identity == 1
        depth = 1+max(calculateDepth(tree.left),calculateDepth(tree.right));
    else
        depth = 0;
    end
    
end