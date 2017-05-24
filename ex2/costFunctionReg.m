function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
h = sigmoid(X*theta);
%theta2 = theta(2, end);
%theta2(1,:) = [];
theta2 = theta;
theta2(1,1) = 0;

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J = (1/m) * ((-1) *y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*sum(theta2 .* theta2) ;

for j = 1:size(theta,1)
  sum = 0;
  for i = 1:m
    sum = sum + ( sigmoid(X(i,:)*theta) - y(i,1) )*X(i,j);
  endfor
  if ( j == 1 )
    grad(j,1) = (1/m)*sum;
  else
    grad(j,1) = (1/m)*sum + (lambda/m)*theta(j,1);
  endif
endfor

% =============================================================

end
