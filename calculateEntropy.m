function entropy = calculateEntropy(data)
    [m,n] = size(data);
    label = data(:,n);
    labelID = unique(label);
    
    numOflabel = length(labelID);
    prob = zeros(numOflabel,2);
    
    if(numOflabel == 1)
        entropy =0;
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
        entropy = 0;
        for i = 1:numOflabel
            entropy = entropy - prob(i,2) * log2(prob(i,2));
        end
    end
end