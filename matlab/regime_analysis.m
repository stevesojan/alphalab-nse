function T = regime_analysis(exportDir, outDir)
%REGIME_ANALYSIS Volatility regimes from a 2-component Gaussian mixture (fitgmdist) and per-regime Sharpe.
%   Market proxy = equal-weighted daily return of the universe (wide_close.parquet). Feature = log of
%   21-day rolling volatility. Each day is labelled low/high volatility by the fitted mixture, then the
%   Sharpe of each study family is computed within each regime. Regime labels use the full sample, so
%   this is descriptive (where did each signal earn?), not a tradable rule.
W = parquetread(fullfile(exportDir, 'wide_close.parquet'));
U = parquetread(fullfile(exportDir, 'wide_in_universe.parquet'));
dates = W{:, 1};
C = W{:, 2:end};
Uu = U{:, 2:end} > 0;
R = [nan(1, size(C, 2)); C(2:end, :) ./ C(1:end-1, :) - 1];
R(~Uu) = NaN;
mkt = mean(R, 2, 'omitnan');
vol = movstd(mkt, [20 0], 'omitnan');
feat = log(vol);
ok = isfinite(feat) & dates >= datetime(2015, 1, 1);
rng(1);
gm = fitgmdist(feat(ok), 2, 'Replicates', 5);
[~, hiComp] = max(gm.mu);
lab = nan(size(feat));
lab(ok) = cluster(gm, feat(ok)) == hiComp;      % 1 = high volatility regime
files = dir(fullfile(exportDir, 'daily_*.parquet'));
rows = cell(numel(files), 5);
for i = 1:numel(files)
    t = parquetread(fullfile(files(i).folder, files(i).name));
    t = t(t.gross > 0, :);
    [tf, loc] = ismember(t.date, dates);
    regime = nan(height(t), 1);
    regime(tf) = lab(loc(tf));
    lo = t.pnl(regime == 0); hi = t.pnl(regime == 1);
    rows(i, :) = {erase(erase(files(i).name, 'daily_'), '.parquet'), sh(lo), sh(hi), numel(lo), numel(hi)};
end
T = cell2table(rows, 'VariableNames', {'family', 'sharpe_low_vol', 'sharpe_high_vol', 'days_low', 'days_high'});
if ~exist(outDir, 'dir'); mkdir(outDir); end
writetable(T, fullfile(outDir, 'regime_sharpe.csv'));
f = figure('Visible', 'off', 'Position', [100 100 900 350]);
plot(dates, vol * sqrt(252), 'k'); hold on;
hiDays = lab == 1;
scatter(dates(hiDays), vol(hiDays) * sqrt(252), 4, 'r', 'filled');
ylabel('annualised 21-day vol, equal-weight universe'); title('Volatility regimes (red = high-vol component)');
exportgraphics(f, fullfile(outDir, 'regimes.png'), 'Resolution', 130);
close(f);
end

function s = sh(x)
if numel(x) > 2 && std(x) > 0; s = mean(x) / std(x) * sqrt(252); else; s = NaN; end
end
