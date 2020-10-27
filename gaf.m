function F = gaf(I,J)
 I = single(I);
 J = single(J);
% I = I./max(I);
% J = J./max(J);

h = [-1 -1 -1;-1 9 -1;-1 -1 -1];
H1 = imfilter(I,h);
H2= imfilter(J,h);

 W1 = sigmoid(H1);
 W2 = sigmoid(H2);
F1 = W1.*I;
F2 = W2.*J;
F = F1 + F2;