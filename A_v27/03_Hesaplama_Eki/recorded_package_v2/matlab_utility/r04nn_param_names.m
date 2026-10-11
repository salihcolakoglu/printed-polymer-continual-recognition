function names = r04nn_param_names(model)
switch model.arch
    case 'linear', names = {'W','b'};
    case 'mlp_tanh', names = {'W1','b1','W2','b2'};
    otherwise, error('r04nn:Architecture','Unknown architecture.');
end
end
