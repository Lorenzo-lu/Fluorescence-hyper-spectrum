function [W] = Gradient_descent(matrix, target, LR,lambda, beta)
%LR: learning rate, beta: momentum, lambda: regularization
%% init the W
[r,c] = size(matrix);
W = zeros(c,1);
VdW = zeros(c,1);

% converge = 1e-4;
converge = mean(target)*1e-7;
% LR = mean(target)/ sum(mean(matrix)) / r * 5e-5;
plot_step = 100;

cost = [];
epoch = 1;
c0 = 1;
c = 0;

%% Loop
% while abs(c-c0) > converge      
while abs(c-c0) > converge    
   
    
    c0 = c;
    
    pred = matrix * W;
    diff = -pred + target;
    c = (diff' * diff)/2/length(diff);

%   

    if mod(epoch,plot_step) == 1
        s = length(cost) + 1;
        cost(s) = c;
        fprintf('cost is %d\t LR is %d\n',c,LR);
    end

    dW = matrix' * diff;

    if epoch == 1
        VdW = dW;  %% momentum
    else
        VdW = beta * VdW + (1-beta) * dW;
    end
    
    delta = (VdW - lambda * sign(W));

    W = W + LR * delta;  
%     W = W - LR * (VdW + lambda * (W));     

    epoch = epoch + 1;
   
end

figure;
plot(cost);
xlabel('step');
ylabel('cost');

end