% function [yy1,iter_num,obj] = SMCOBG(X,para,k)
function [yy1,iter_num,obj] = SMCOBG(X, W, para, k)

%% 初始化
%保证所有 X{v} 为 d × n
% for v = 1:length(X)
%     if size(X{v},1) > size(X{v},2)
%         % 当前是 n × d，转置
%         X{v} = X{v}';
%     end
% end


alpha = para.alpha;
gamma = para.gamma;
self_q =  para.self_q;
cross_q = para.cross_q;
m=para.m;

c = para.c;
d = para.d;
n = size(X{1},2);




MaxIter = 50;
n_view = length(X);


[X] = data_prep(X);  % 数据预处理


%% 生成二部图


A = cell(d,m);  % 初始化A
C = cell(1,n_view);
S = cell(1,n_view);  % 初始化为 cell
q = cross_q*ones(n_view) - diag(cross_q*ones(1,n_view)) + diag(self_q*ones(1,n_view));
q_coef = q + eye(n_view);
Q = cell(n_view,1);
D = cell(1,n_view);
for v = 1:n_view
    S{v} =  W{v}; 
end

if nnz(S{1})/numel(S{1}) < 0.4  % if W contains a large proportion of zeros, use sparse mode
    for i=1:v
        S{i} = sparse(S{i});
    end
    sparse_mode = true;
else
    for i=1:v
        S{i} = full(S{i});
    end
    sparse_mode = false;
end
%确保抑制部分不会过拟合，不构成虚边
if sparse_mode
    SSS = sparse(m,n);
else
    SSS = zeros(m,n);
end

for i=1:n_view
    SSS = max(SSS, S{i});
    C{i} = full(S{i});
end
for i=1:n_view
    if sparse_mode
        C{i} = sparse(C{i});
    end
    C{i} = min(C{i}, SSS);
end

ww = 1:2^n_view-2; % beta can't be all zeros, so -2
logww = log2(ww);
zz = 2.^(0:n_view-1);
yy = ww(abs(floor(logww)-logww)>eps);
beta_zeros_ones = de2bi([0,zz,yy]);
n_eye_coef = -eye(n_view);
beta  = ones(1,n_view) / n_view;


P = zeros(m,n);
for v = 1:n_view
    P = P+beta(v)*C{v};  
end
P = P/n_view;
for j = 1:n
    if sum(P(:,j)) ~= 0
        P(:,j) = P(:,j)/sum(P(:,j));
    end

end





%% 开始迭代求解
obj = zeros(MaxIter,1);

for iter = 1:MaxIter


    
   
    S_pre = S;
    A_pre = A;
    C_pre = C;
    P_pre = P;
   



    %% 1.更新A
    B = cell(d,m);
    for v = 1:n_view
        B{v} = X{v}*S{v}';
    end
    for v = 1:n_view
        [U_c,~,V_c] = svd(B{v}, "econ");
        A{v} = U_c * V_c';  % 每个视角自己的A
    end
  
    %% 2.更新beta
    obj1 = 0;
    for v =1:n_view
        D{v} = S{v} - C{v};
        temp = norm(beta(v)*C{v}-P, 'fro');
        obj1 = obj1 + temp*temp;
    end
    coef = zeros(v);
    coef2 = zeros(v);
    
    for i=1:n_view%wss
        for j=i: n_view%wss
            coef(i,j) = sum(sum(C{i}.*C{j}));
            coef(j,i) = coef(i,j);
            coef2(i,j) = sum(sum(D{i}.*D{j}));
            coef2(j,i) = coef2(i,j);
        end
    end
    coef2 = coef2 .* q;



    % 构造线性方程
    f = zeros(n_view, 1);
 
    K = diag(diag(coef));
    for v = 1:n_view
        f(v)= trace(C{v} * P');
    end
    KK = K + coef2;
    one = ones(v, 1);
  

    for i=1:1
        mpl = beta_zeros_ones(i,:);
        coef3 = KK .* ~mpl + n_eye_coef .* mpl;
        JJ = [coef3, one; 1-mpl, 0];
        temp_b = [f; 1];
        if det(JJ) == 0 % abs(det(X)) <= eps
      
            solution = temp_b*pinv(JJ);%wss
         
        else
            solution = JJ \ temp_b; 
           
        end
       beta = EProjSimplex_new(solution(1:v));
    end

    %% 3.更新C
    alp_coef = beta * beta';
   ccoef = alp_coef .* q_coef;
   
    if sparse_mode
        commom_baW = sparse(m, n);
    else
        commom_baW = zeros(m,n);
    end
    for i=1:n_view
        baW{i} = cross_q*beta(i)*S{i};
        special_baW{i} = self_q*beta(i)*S{i};
        commom_baW = commom_baW + baW{i};
    end

    for i=1:n_view
        true_baW{i} = commom_baW-baW{i}+special_baW{i};
        temp = full(beta(i)*(P + true_baW{i}));
        Q{i} = temp;
    end
    right_q = zeros(n_view, m*n);
    for i = 1:n_view
        right_q(i,:) = Q{i}(:)';
    end

    if det(ccoef) == 0
        solution1 = (pinv(ccoef) * right_q)';
        
    else
        solution1 = (ccoef \ right_q)';
        
    end
    solution1(solution1<0) = 0;
    C_change = 0;
    for i=1:n_view
        temp = solution1(:,i);
        oldC = C{i};
        C{i} = zeros(m, n);
        C{i} = reshape(temp, m, n);
        
        C{i} = min(S{i}, C{i});
        if sparse_mode
            C{i} = sparse(C{i});
        end
        
    end
if any(isnan(C{1}(:))) || any(isinf(C{1}(:)))
    error('C contains NaN or Inf values after update!');
end
    %% 4.更新S
    commom_baE = zeros(m,n);
        for i = 1:n_view 
            baE{i} = cross_q*beta(i)*(S{i}-C{i});
            special_baE{i} = self_q*beta(i)*(S{i}-C{i});
            commom_baE = commom_baE + baE{i};
        end
        for i = 1:n_view 
            true_baE{i} = commom_baE - baE{i} + special_baE{i};
            tem = full(beta(i)*true_baE{i});
            Q{i} = tem;
            
        end

        for v = 1:n_view
            YY = (X{v}'*A{v}+alpha*W{v}'-0.5* Q{v}')./(1+alpha);
            
            YYYY = zeros(size(YY));
            for is = 1:n
               
                YYYY(is,:) = EProjSimplex_new(max(YY(is,:),0));
            end
            S{v} = YYYY';
        end
        
if any(isnan(S{1}(:))) || any(isinf(S{1}(:)))
    error('S contains NaN or Inf values after update!');
end
   
    %% 5.更新 F
   % 检查 P 矩阵
    epsilon = 1e-12;         % 很小的正数，防止0
    row_sum = sum(P,2);      
    row_sum(row_sum < epsilon) = epsilon; 
    Dm = diag(row_sum.^(-0.5));    
    Dn = diag(sum(P).^(-0.5));
    
    if any(isnan(Dm(:))) || any(isinf(Dm(:)))
    error('Dm contains NaN or Inf before multiplication!');
end
    XX = Dm*P*Dn;
    if any(isinf(XX), 'all') || any(isnan(XX), 'all')
        XX(isinf(XX)) = 0;
        XX(isnan(XX)) = 0;
    end
    
    if any(isnan(XX(:))) || any(isinf(XX(:)))
    error('XX contains NaN or Inf');
end
    
    [U,SS,V] = svds(XX,c+1);
    
    if any(isnan(U(:))) || any(isinf(U(:))) || any(isnan(SS(:))) || any(isinf(SS(:))) || any(isnan(V(:))) || any(isinf(V(:)))
    error('SVD results contain NaN or Inf');
    end

    Fn = sqrt(2)*U(:,1:c)/2;
    Fm = sqrt(2)*V(:,1:c)/2;
   
    if any(isnan(Fn(:))) || any(isinf(Fn(:))) || any(isnan(Fm(:))) || any(isinf(Fm(:)))
    error('Fn or Fm contains NaN or Inf');
end

    ev = diag(SS);
    
    Fn_old = Fn;
    Fm_old = Fm;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 < c-0.0000001
        gamma = 2*gamma;
    elseif fn2 > c+1-0.0000001
        gamma = gamma/2;   Fn = Fn_old; Fm = Fm_old;
    % else
    %     break;
    end



    %% 6.更新 P
    % Compute G
    P_pre = P;
    G = zeros(m,n); 
    for v = 1:n_view
            G= G + beta(v)*C{v};
    end
    G = G/n_view;
    Fn_ = Dm*Fn;
    Fm_ = Dn*Fm;
    dist = L2_distance_1(Fn_',Fm_');
    P = zeros(m,n);
    for i = 1:n
        gi = G(:,i);
        ti = dist(:,i);
        ad = gi-0.5*gamma*ti;
        P(:,i) = EProjSimplex_new(ad);
    end
    %% 8.记录obj
    tol = 1e-6; 
    change_S = 0;

    change_C = 0;

    for v = 1:n_view
        change_S = change_S + norm(S{v} - S_pre{v}, 'fro');
    
        change_C = change_C + norm(C{v} - C_pre{v}, 'fro');
    end
        change_P = norm(P - P_pre, 'fro');


        

if iter > 3
    if change_S < tol &&  change_C < tol && change_P < tol
        disp('Converged!');
        break;
    end
end



end
fprintf("iter=%d\n", iter);
%% 9.生成增广图M

M = sparse(n+m,n+m);
M(1:n,n+1:end) = P';
M(n+1:end,1:n) = P;
M = (M + M') / 2;  
GG = graph(M);  % 创建图对象，其中 adjacencyMatrix 是你的邻接矩阵
yyy = conncomp(GG);  % bins 返回一个数组，表示每个节点所属的连通组件
yy1 = yyy(1:n)';

% 返回迭代次数
iter_num = iter;  

end

function [X] = data_prep(X)
% 数据预处理函数
%Input:
%       X: cell类型,数据矩阵, size(X{i})=d*n;
%Output:
%       X: cell类型,数据矩阵, size(X{i})=d*n;

v = length(X);
for i = 1:v
    X{i} = X{i} ./ (repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);  % 单位向量归一化
end

end



