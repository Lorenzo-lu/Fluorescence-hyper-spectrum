function [W]=Ridge_R(A,d,lambda)

[m,n]=size(A);
number=length(lambda);

for i=1:number
    W(:,i)=A'*inv(A*A'+lambda(i)*eye(m))*d;
end

end