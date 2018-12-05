function [G, gamma] = FISTA_gLASSO_backtracking(G_init, gamma_init, triplets, label, weight, lambda, no_dim, opts)

% * A Fast Iterative Shrinkage-Thresholding Algorithm for 
% Linear Inverse Problems: FISTA (backtracking version)

% * Solve the problem: 
%   G, gamma = arg min_{G, gamma} F(G, gamma) = f(G) + lambda * g(gamma) where:
%   - G: variable
%   - gamma: variable
%   - f(G): a smooth convex function with continuously differentiable 
%       with Lipschitz continuous gradient `L(f)` (Lipschitz constant of 
%       the gradient of `f`).
%   - lambda: regularization parameter
%   - g(gamma): a non-smooth convex regularization for sparsity

%  INPUT:
%       calc_f    : a function calculating f(G) in F(G, gamma) = f(G) + g(gamma) 
%       grad_G    : a function calculating gradient of f(G) given G.
%       G_init    : a matrix -- initial value.
%       gamma_init: a vector -- initial value
%       opts      : a structure
%           opts.lambda  : a regularization parameter, can be either a scalar or
%                          a weighted matrix.
%           opts.max_iter: maximum iterations of the algorithm. 
%                           Default 300.
%           opts.tol     : a tolerance, the algorithm will stop if difference 
%                           between two successive X is smaller than this value. 
%                           Default 1e-8.
%           opts.verbose : showing F(X) after each iteration or not. 
%                           Default false. 
%           opts.L0 : a positive scalar. Initial Lipschitz constant
%           opts.eta: (must be > 1). eta in the algorithm (page 194)
%       calc_F    : optional, a function calculating value of F at G, gamma 
%               via feval(calc_F, X, gamma). 

%  OUTPUT:
%      G, gamma   : solution
% ************************************************************************
% * Date created    : 14/08/18
% * Author          : Ke Ma 
% * Date modified   : 
% ************************************************************************


	if ~isfield(opts, 'max_iter')
		opts.max_iter = 300;
	end
	
	if ~isfield(opts, 'tol')
		opts.tol = 1e-10;
	end
	
	if ~isfield(opts, 'verbose')
		opts.verbose = false;
	end

	if ~isfield(opts, 'L0')
		opts.L0 = 1;
	end 

	if ~isfield(opts, 'eta')
		opts.eta = 2;
	end

	N = size(G_init, 1);
	M = size(triplets, 1);

	x_old = gamma_init .* weight;
	y_old = gamma_init .* weight;
	G_old = G_init;
	K_old = G_init;
	t_old = 1;
	iter = 0;
	cost_old = 1e10;

	% MAIN LOOP
	L = opts.L0;
	while  iter < opts.max_iter
		iter = iter+1;
		[grad_G, grad_beta, disper] = grad_cal(K_old, y_old, triplets, label, weight, N, M);
		Lbar = L;
		%%%%%%%%%%%%%%%%%%%% backtracking %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% find i_k 
		while true
			[F, Q] = eval_FQ(K_old, y_old, grad_G, grad_beta, disper, triplets, label, weight, no_dim, lambda, Lbar, M);
			if F <= Q
				break;
			end
			Lbar = Lbar * opts.eta;
			L = Lbar;
		end
		x_new = proximal_beta(y_old, grad_beta, lambda, L);
		t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
		y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
		G_new = proj_grad_G(G_old, grad_G, no_dim, L);
		K_new = G_new + (t_old - 1)/t_new * (G_new - G_old);
		% check stop criteria
% 		e = norm(x_new - x_old, 1)/numel(x_new);
% 		if e < opts.tol
% 			break;
% 		end
		% update
		x_old = x_new;
		t_old = t_new;
		y_old = y_new;
		G_old = G_new;
		K_old = K_new;
		%% show progress
		if opts.verbose
			if nargin ~= 0
				loss_value = f_eval(K_old, triplets, y_old);
				f_val = dot(weight, loss_value.^2);
				g_val = g_eval(y_old, weight, lambda);
				cost_new = f_val + g_val;
				if cost_new <= F 
					stt = 'Yes.';
				else 
					stt = 'NO, check your code.';
				end
				fprintf('iter = %3d, cost = %f, cost decreases? %s\n', ...
					iter, cost_new, stt);
				cost_old = cost_new;
			else 
				if mod(iter, 5) == 0
					fprintf('.');
				end
				if mod(iter, 10) == 0 
					fprintf('%d', iter);
				end
			end
		end
		D = bsxfun(@plus, bsxfun(@plus, -2 .* G_new, diag(G_new)), diag(G_new)');
		no_test_viol = 0;
		for m = 1:M
			if label(m) == 1 && D(triplets(m, 1), triplets(m, 2)) > D(triplets(m, 1), triplets(m, 3))
				no_test_viol = no_test_viol + 1;
			elseif label(m) == -1 && D(triplets(m, 1), triplets(m, 2)) < D(triplets(m, 1), triplets(m, 3))
				no_test_viol = no_test_viol + 1;
			end
		end
		test_goe = no_test_viol ./ M;
		fprintf('Training Error: %f\n', test_goe);
	end
	G = G_new;
	gamma = x_new ./ weight;
end

%% grad_cal: Calculate Partial Gradient for Gram matrix
function [grad_G, grad_beta, disper] = grad_cal(G, beta, triplets, label, weight, N, M)
	mu_val = zeros(M, 1);
	disper = zeros(M, 1);
	grad_G = zeros(N, N);
	for i = 1:M
		mu_val(i) =  G(triplets(i, 1), triplets(i, 3)) + G(triplets(i, 3), triplets(i, 1)) - G(triplets(i, 1), triplets(i, 2)) ...
			- G(triplets(i, 2), triplets(i, 1)) - G(triplets(i, 3), triplets(i, 3)) + G(triplets(i, 2), triplets(i, 2));

		disper(i) = weight(i) * mu_val(i) + beta(i) - weight(i) * label(i);	

		grad_G(triplets(i, 1), triplets(i, 2)) = grad_G(triplets(i, 1), triplets(i, 2)) - disper(i);
		grad_G(triplets(i, 2), triplets(i, 1)) = grad_G(triplets(i, 2), triplets(i, 1)) - disper(i);

		grad_G(triplets(i, 1), triplets(i, 3)) = grad_G(triplets(i, 1), triplets(i, 3)) + disper(i);
		grad_G(triplets(i, 3), triplets(i, 1)) = grad_G(triplets(i, 3), triplets(i, 1)) + disper(i);

		grad_G(triplets(i, 2), triplets(i, 2)) = grad_G(triplets(i, 2), triplets(i, 2)) + disper(i);
		grad_G(triplets(i, 3), triplets(i, 3)) = grad_G(triplets(i, 3), triplets(i, 3)) - disper(i);
	end
	grad_beta = disper ./ weight;
end

%% proj_grad_G: Projected Gradient Descent on G
function [G_new] = proj_grad_G(G, grad_G, no_dim, L)
	G_new = G - (1 / L) * grad_G;
	% Project Gram matrix G back onto the PSD cone
	[V, Sigma] = eig(G_new);
	V = real(V);
	Sigma = real(Sigma);
	ind = find(diag(Sigma) > 0);
	if isempty(ind)
		warning('Projection onto PSD cone failed. All eigenvalues were negative.'); 
	end
	if length(ind) > no_dim
		G_new = V(:, ind(1:no_dim)) * Sigma(ind(1:no_dim), ind(1:no_dim)) * V(:, ind(1:no_dim))';
	else
		G_new = V(:, ind) * Sigma(ind, ind) * V(:, ind)';
	end
	if any(isinf(G_new(:)))
		warning('Projection onto PSD cone failed. Metric contains Inf values.'); 
	end
	if any(isnan(G_new(:)))
		warning('Projection onto PSD cone failed. Metric contains NaN values.'); 
	end
end

%% proximal_beta: Proximal Gradient Descent on beta
function [beta_new] = proximal_beta(beta, grad_beta, lambda, L)
	hat_beta = beta - 1 / L * grad_beta;
	beta_new = max(abs(hat_beta) - lambda / L, 0) .* sign(hat_beta);
end


%% cal_Q: function description
function [FL, QL] = eval_FQ(G_1, beta_1, grad_G, grad_beta, disper, triplets, label, weight, no_dim, lambda, L, M)
	mu_val = zeros(M, 1);
	disper_2 = zeros(M, 1);
	G_2 = proj_grad_G(G_1, grad_G, no_dim, L);
	beta_2 = proximal_beta(beta_1, grad_beta, lambda, L);
	tmp_G = G_2 - G_1;
	tmp_G = tmp_G(:);
	tmp_beta = beta_2 - beta_1;
	f = 0.5 * sum(disper.^2 ./ weight);
	QL = f + sum(sum(grad_G(:) .* tmp_G)) + dot(grad_beta, tmp_beta) + 0.5 * L * sum(tmp_G.^2) + ...
		0.5 * L * sum(tmp_beta.^2);
	for i = 1:M
		mu_val(i) =  G_2(triplets(i, 1), triplets(i, 3)) + G_2(triplets(i, 3), triplets(i, 1)) - G_2(triplets(i, 1), triplets(i, 2)) ...
			- G_2(triplets(i, 2), triplets(i, 1)) - G_2(triplets(i, 3), triplets(i, 3)) + G_2(triplets(i, 2), triplets(i, 2));

		disper_2(i) = weight(i) * mu_val(i) + beta_2(i) - weight(i) * label(i);	
	end
	FL = 0.5 * sum(disper_2.^2 ./ weight);
end