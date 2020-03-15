clear
close all
clc
%%
%% load data
pixel_edge = 100;
raw = im2double(imread('ECE.jpg'));
% A = (A(:,:,1)+A(:,:,2)+A(:,:,3))/3;
raw = imresize(raw,[pixel_edge,pixel_edge]);

recons = zeros(size(raw));

idct_matrix = inv(dctmtx(pixel_edge));

basis = kron(idct_matrix,idct_matrix');

% basis = (dctmtx(pixel_edge^2));


tic;
% measure_mat = round(rand(round(pixel_edge^2 * 1),pixel_edge^2)); 
measure_mat = FT_mask(pixel_edge, round(pixel_edge^2 * 0.2));
% measure_mat = ceil(basis(1:round(pixel_edge^2 * 0.2),:));

for i = 1:3
    A = raw(:,:,i);
    img = reshape(A,[pixel_edge^2,1]);
    measure = measure_mat * img;

    Theta = measure_mat * basis;
    lambda = 0.1;
%     w = ista_solve_hot( Theta, measure, lambda );
%     w = Ridge_R(Theta,measure,lambda);
    w = Gradient_descent(Theta, measure, 1e-6, lambda, 0.5);

    rec = basis * w;
    recons(:,:,i) = reshape(rec,[pixel_edge,pixel_edge]);
end
toc;
%rec = rec/max(rec);
figure;
imshow(recons);
figure;
imshow(raw);