function [M] = FT_mask(pixel_edge, cycles)

% This function is used to generalize a mask matrix by an idea similar to
% Fourier Transform
% The size of mask generalized by IDCT2 function can be described as a lot
% of pads from the big to the smaller.
% In the first step, all the elements are 1
% In the next step, only the left half are 1 ...
% In this way, all the vectors are linear independent
% The size of the pads from the big to the smaller can make the image
% reconstructed from the outline to the details. But in most scenarios, the
% details are less important. -> [This can contribute to a sparsity!]




i = 1;
diag_i = 1;
while i <= cycles
    index = freq_select_2d(diag_i);
    j = 1;
    while j <= (2 * diag_i - 1)
        row = index(j,1);
        col = index(j,2);
        j = j + 1;
        
        a = zeros([pixel_edge,pixel_edge]);
        a(row,col) = 1;
        V = ceil(idct2(a));
        
%         V(find(V<0)) = 0;
%         V(find(V>0)) = 1;
        V = reshape(V,[1,pixel_edge^2]);    
        M(i,:) = V; 
        
        i = i + 1;
        if i > cycles
            break;
        end
    end
    diag_i = diag_i + 1;
    
end

%% this is dct matrix it seems not a good idea
% M = dctmtx(pixel_edge^2);
% M = ceil(M);
% M = M(1:cycles,:);

end