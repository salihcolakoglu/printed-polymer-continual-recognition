function P = a24_sigmoid(Z)
% Stable unsoftened sigmoid. No task information or labels enter this map.
validateattributes(Z,{'double'},{'real','finite'});
P = zeros(size(Z));
nonnegative = Z>=0;
P(nonnegative) = 1./(1+exp(-Z(nonnegative)));
e = exp(Z(~nonnegative));
P(~nonnegative) = e./(1+e);
end
