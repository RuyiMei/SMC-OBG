function [W, BB, centers] = ConstructBipartiteGraph(X, m, k)

n_view = length(X);
n = size(X{1},2);

BB = cell(n_view,1);
centers = cell(n_view,1);
opts.style = 4; %
%% ================= Anchor Selection =================

if opts.style == 1     % direct sample

    XX = [];
    for v = 1:n_view
        XX = [XX; X{v}];
    end

    [~,ind,~] = graphgen_anchor(XX,m);

    for v = 1:n_view
        centers{v} = X{v}(:,ind);
    end

elseif opts.style == 2 % random sample

    vec = randperm(n);
    ind = vec(1:m);

    for v = 1:n_view
        centers{v} = X{v}(:,ind);
    end

elseif opts.style == 3 % KNP

    XX = [];
    for v = 1:n_view
        XX = [XX; X{v}];
    end

    [~,~,~,~,dis] = litekmeans(XX',m);

    [~,ind] = min(dis,[],1);
    ind = sort(ind,'ascend');

    for v = 1:n_view
        centers{v} = X{v}(:,ind);
    end

elseif opts.style == 4 % kmeans sample

    XX = [];
    len = zeros(n_view,1);

    for v = 1:n_view
        XX = [XX; X{v}];
        len(v) = size(X{v},1);
    end

    [~,Cen,~,~,~] = litekmeans(XX',m);

    Cen = Cen';

    t1 = 1;

    for v = 1:n_view

        t2 = t1 + len(v) - 1;

        centers{v} = Cen(t1:t2,:);

        t1 = t2 + 1;

    end

end

%% ================= Bipartite Graph =================

for v = 1:n_view

    DD = L2_distance_1(X{v}, centers{v});

    [~,idx] = sort(DD,2);

    BB{v} = zeros(n,m);

    for i = 1:n

        id = idx(i,1:k+1);

        di = DD(i,id);

        BB{v}(i,id) = ...
            (di(k+1)-di) ./ ...
            (k*di(k+1)-sum(di(1:k))+eps);

    end

end

%% ================= W =================

W = cell(1,n_view);

for v = 1:n_view
    W{v} = BB{v}';
end

end