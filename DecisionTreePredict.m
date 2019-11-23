function label = DecisionTreePredict(tree,data)
if tree.identity ~= 2
    if tree.feature == 1
        if data(1) < tree.value
            label = DecisionTreePredict(tree.left,data);
        else
            label = DecisionTreePredict(tree.right,data);
        end
    else
        if data(2) < tree.value
            label = DecisionTreePredict(tree.left,data);
        else
            label = DecisionTreePredict(tree.right,data);
        end
    end
else
    label = tree.label;
end
end