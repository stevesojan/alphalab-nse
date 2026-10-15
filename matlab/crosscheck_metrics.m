function T = crosscheck_metrics(exportDir, outDir)
%CROSSCHECK_METRICS Independent MATLAB re-implementation of the six metrics.
%   Reads every daily_*.parquet written by the Python study (columns date, pnl, traded, ...), restricts
%   to dates with positions, computes Sharpe, Turnover, Fitness, Returns, Drawdown, Margin with the same
%   definitions as alphalab/backtest/metrics.py, and writes outDir/crosscheck_metrics.csv. The Python
%   side (alphalab.matlab_check) compares the two within 1e-9.
files = dir(fullfile(exportDir, 'daily_*.parquet'));
names = strings(numel(files), 1);
M = zeros(numel(files), 7);
for i = 1:numel(files)
    t = parquetread(fullfile(files(i).folder, files(i).name));
    t = t(t.gross > 0, :);                     % same filter as pipeline.run
    m = six_metrics(t.pnl, t.traded, 0.5);
    names(i) = erase(erase(files(i).name, 'daily_'), '.parquet');
    M(i, :) = [m.sharpe, m.turnover, m.fitness, m.returns, m.drawdown, m.margin_bps, numel(t.pnl)];
end
T = array2table(M, 'VariableNames', {'sharpe', 'turnover', 'fitness', 'returns', 'drawdown', 'margin_bps', 'days'});
T = addvars(T, names, 'Before', 1, 'NewVariableNames', 'family');
if ~exist(outDir, 'dir'); mkdir(outDir); end
writetable(T, fullfile(outDir, 'crosscheck_metrics.csv'));
end

function m = six_metrics(pnl, traded, halfBook)
D = 252;
s = std(pnl, 0);                               % ddof = 1
if s > 0; m.sharpe = mean(pnl) / s * sqrt(D); else; m.sharpe = 0; end
m.returns = mean(pnl) * D / halfBook;
m.turnover = mean(traded);
cum = cumsum(pnl);
peak = cummax([0; cum]);
m.drawdown = max(peak(2:end) - cum) / halfBook;
m.margin_bps = sum(pnl) / sum(traded) * 1e4;
m.fitness = m.sharpe * sqrt(abs(m.returns) / max(m.turnover, 0.125));
end
