clear; clc; close all;
load fisheriris
classes = unique(species);

% Use principal component analysis to reduce the dimension of the data to two dimensions for visualization.
[~,score] = pca(meas(:,3:4),'NumComponents',1);

GMModels = cell(3,1); % Preallocation
options = statset('MaxIter',1000);
rng(1); % For reproducibility

for j = 1:4
    GMModels{j} = fitgmdist(score,j,'Options',options);
    fprintf('\n GM Mean for %i Component(s)\n',j)
    Mu = GMModels{j}.mu;
end

figure;
for j = 1:4
    subplot(2,2,j)
    h1 = gscatter(score,zeros(size(score)),species);
    h = gca;
    hold on
    gmPDF = @(x) arrayfun(@(x0) pdf(GMModels{j},x0),x);
    fplot(gmPDF, [h.XLim(1), h.XLim(2)]);
    % fcontour(gmPDF,[h.XLim h.YLim],'MeshDensity',100)
    title(sprintf('GM Model - %i Component(s)',j));
    xlabel('1st principal component');
    ylabel('2nd principal component');
    if(j ~= 3)
        legend off;
    end
    hold off
end
g = legend(h1);
g.Position = [0.7 0.25 0.1 0.1];


