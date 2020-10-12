function [selection] = freq_select_2d(i)
% here the matrix will give a series of index composed of  [i i; i-1 i; i
% i-1;...]

j = 1;
v_up = [];
v_left = [];
while j < i
    v_left(j,1) = i ;
    v_left(j,2) = i - j;
    v_left(j,3) = 2 * j - 1;
    
    v_up(j,1) = i - j;
    v_up(j,2) = i ;
    v_up(j,3) = 2 * j;
    
    j = j + 1;
end

v = [v_up ; v_left];

v(2*i-1,1) = i;
v(2*i-1,2) = i;
v(2*i-1,3) = 0;

v = sortrows(v,3);

selection = v(:,1:2);
end
