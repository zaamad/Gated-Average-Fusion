function y = sigmoid(x,c,a)


narginchk(1,3) 
if nargin<3
    a = 1; 
else
    assert(isscalar(a)==1,'a must be a scalar.') 
end

if nargin<2
    c = 0; 
else
    assert(isscalar(c)==1,'c must be a scalar.') 
end

%% Perform mathematics: 

y = 1./(1 + exp(-a.*(x-c)));
