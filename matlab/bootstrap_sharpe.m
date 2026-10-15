function T = bootstrap_sharpe(exportDir, outDir, nBoot)
%BOOTSTRAP_SHARPE Sharpe confidence intervals with the Statistics and Machine Learning Toolbox.
%   For each daily_*.parquet: (a) bootci percentile interval treating days as i.i.d., and (b) a moving-
%   block bootstrap (block 20 days) to respect autocorrelation. Python reports a stationary bootstrap;
%   the three should be close for weakly autocorrelated PnL and diverge when serial correlation matters.
if nargin < 3; nBoot = 2000; end
rng(1);
files = dir(fullfile(exportDir, 'daily_*.parquet'));
rows = cell(numel(files), 6);
sharpeFn = @(x) mean(x) / std(x) * sqrt(252);
for i = 1:numel(files)
    t = parquetread(fullfile(files(i).folder, files(i).name));
    x = t.pnl(t.gross > 0);
    ci = bootci(nBoot, {sharpeFn, x}, 'Type', 'per');
    b = block_boot(x, nBoot, 20, sharpeFn);
    rows(i, :) = {erase(erase(files(i).name, 'daily_'), '.parquet'), sharpeFn(x), ci(1), ci(2), ...
                  prctile(b, 2.5), prctile(b, 97.5)};
end
T = cell2table(rows, 'VariableNames', {'family', 'sharpe', 'iid_lo', 'iid_hi', 'block_lo', 'block_hi'});
if ~exist(outDir, 'dir'); mkdir(outDir); end
writetable(T, fullfile(outDir, 'bootstrap_sharpe.csv'));
end

function s = block_boot(x, nBoot, L, fn)
n = numel(x);
nb = ceil(n / L);
s = zeros(nBoot, 1);
for b = 1:nBoot
    starts = randi(n - L + 1, nb, 1);
    idx = reshape(starts + (0:L-1), [], 1);
    s(b) = fn(x(idx(1:n)));
end
end
