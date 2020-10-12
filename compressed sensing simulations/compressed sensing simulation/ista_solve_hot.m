function X = ista_solve_hot( A, d, lambda )
%ista_solve_hot: Iterative soft-thresholding for multiple values of
%lambda with hot start for each case - the converged value for the previous
%value of lambda is used as an initial condition for the current lambda.
%   this function solves the minimization problem
%   Minimize |Ax-d|_2^2 + lambda*|x|_1   (Lasso regression)
%   using iterative soft-thresholding.
	
	MAX_ITER = 1e6;         % maximum number of iterations
	TOL = 1e-5;             % convergence tolerance
	
	tau = 1/norm(A)^2;      % choose stepsize
	
	n = size(A,2);
	w = zeros(n,1);         % start point for the iteration
    
    num_lam = length(lambda);
    
    for j = 1:num_lam
	
        for i = 1:MAX_ITER                   
            z = w - tau*(A'*(A*w-d));                        % Landweber
            wold = w;                                        % store old x
            w = sign(z).*max(abs(z)-tau*lambda(j)/2, 0 );  % soft thresh
            if norm(w-wold) < TOL
                break
            end
        end
        
        X(:,j) = w;
    end
    
end
