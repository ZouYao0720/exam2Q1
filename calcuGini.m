function gini = calcuGini(data)
[m,n] = size(data);
    label = data(:,n);
    labelID = unique(label);
    
    numOflabel = length(labelID);
    prob = zeros(numOflabel,2);
    
    if(numOflabel == 1)
        gini =0;
    else
        for i = 1:numOflabel
            prob(i,1) = labelID(i);
            for j =1:m
                if prob(i,1) == label(j)
                    prob(i,2) = prob(i,2)+1;
                end
            end
        end
        
        % calculate entropy
        prob(:,2) = prob(:,2)./m;
        gini = 0;
        for i = 1:numOflabel
            gini = gini + prob(i,2) * (1-prob(i,2));
        end
    end
end