function param_surface(studyDir, outDir)
%PARAM_SURFACE Plot the EMA-cross in-sample Sharpe surface from the Python trial ledger.
L = parquetread(fullfile(studyDir, 'ledger.parquet'));
L = L(L.family == "ema_cross" & L.period == "IS 2015-2022", :);
fast = zeros(height(L), 1); slow = fast;
for i = 1:height(L)
    p = jsondecode(L.params{i});
    fast(i) = p.fast; slow(i) = p.slow;
end
fv = unique(fast); sv = unique(slow);
Z = nan(numel(fv), numel(sv));
for i = 1:height(L)
    Z(fv == fast(i), sv == slow(i)) = L.sharpe(i);
end
f = figure('Visible', 'off', 'Position', [100 100 520 420]);
imagesc(Z); colormap(f, parula); colorbar;
set(gca, 'XTick', 1:numel(sv), 'XTickLabel', sv, 'YTick', 1:numel(fv), 'YTickLabel', fv);
xlabel('slow span'); ylabel('fast span'); title('EMA cross: IS Sharpe (MATLAB, from ledger)');
for i = 1:numel(fv)
    for j = 1:numel(sv)
        text(j, i, sprintf('%.2f', Z(i, j)), 'HorizontalAlignment', 'center', 'Color', 'w');
    end
end
if ~exist(outDir, 'dir'); mkdir(outDir); end
exportgraphics(f, fullfile(outDir, 'ema_surface_matlab.png'), 'Resolution', 130);
close(f);
end
