function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Hypotheses
hx = sigmoid(X * theta);

%%The cost without regression
J_partial = (-y' * log(hx) - (1 - y)' * log(1 - hx)) ./ m;


%%Regression Cost Added
J_regression = (lambda/(2*m)) * sum(theta(2:end).^2);

%%Cost when we add regression
J = J_partial + J_regression;

%Grad without regression
grad_partial = (1/m) * (X' * (hx -y));

%%Grad Cost Added
grad_regression = (lambda/m) .* theta(2:end);

grad_regression = [0; grad_regression];

grad = grad_partial + grad_regression;

% =============================================================

end
