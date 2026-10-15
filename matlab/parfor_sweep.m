function T = parfor_sweep(exportDir, outDir, fasts, slows)
%PARFOR_SWEEP EMA-cross parameter sweep with a simplified, fully specified backtest, serial vs parfor.
%   Simplified pipeline (identical definition in alphalab/matlab_parity.py, used to check the numbers):
%     ema with NaN-skipping recursion; alpha = ema_f/ema_s - 1 inside the universe; cross-sectional
%     demean; weights scaled to gross 1; delay 1: pnl(t) = sum(w(t-2) .* r(t)); traded(t) = sum|w(t-1) - w(t-2)|;
%     Sharpe over dates >= 2015-01-01 with positions.
if nargin < 3; fasts = 2:2:40; end
if nargin < 4; slows = 50:25:300; end
W = parquetread(fullfile(exportDir, 'wide_close.parquet'));
U = parquetread(fullfile(exportDir, 'wide_in_universe.parquet'));
dates = W{:, 1};
C = W{:, 2:end};
Uu = U{:, 2:end} > 0;
R = [nan(1, size(C, 2)); C(2:end, :) ./ C(1:end-1, :) - 1];
R(isnan(R)) = 0;
keep = dates >= datetime(2015, 1, 1);
spans = unique([fasts, slows]);
E = cell(max(spans), 1);
for s = spans; E{s} = ema_nan(C, s); end            % shared EMAs (cheap), sweep cost is in the backtests
[FF, SS] = ndgrid(fasts, slows);
n = numel(FF);
sr1 = zeros(n, 1);
tic;
for k = 1:n
    sr1(k) = sim_one(E{FF(k)}, E{SS(k)}, Uu, R, keep);
end
tSerial = toc;
p = gcp('nocreate');
tPool = 0;
if isempty(p)
    tic; p = parpool('Processes'); tPool = toc;
end
sr2 = zeros(n, 1);
Ef = E;  % broadcast
tic;
parfor k = 1:n
    sr2(k) = sim_one(Ef{FF(k)}, Ef{SS(k)}, Uu, R, keep);
end
tPar = toc;
assert(max(abs(sr1 - sr2)) < 1e-12, 'parfor result differs from serial');
T = table(FF(:), SS(:), sr1, 'VariableNames', {'fast', 'slow', 'sharpe'});
if ~exist(outDir, 'dir'); mkdir(outDir); end
writetable(T, fullfile(outDir, 'parfor_sweep.csv'));
timing = table(n, tSerial, tPar, tPool, p.NumWorkers, 'VariableNames', ...
    {'n_configs', 'serial_s', 'parfor_s', 'pool_start_s', 'workers'});
writetable(timing, fullfile(outDir, 'parfor_timing.csv'));
end

function s = sim_one(ef, es, Uu, R, keep)
a = ef ./ es - 1;
a(~Uu | ~isfinite(a)) = NaN;
a = a - mean(a, 2, 'omitnan');
g = sum(abs(a), 2, 'omitnan');
w = a ./ g;
w(~isfinite(w)) = 0;
T = size(w, 1);
pnl = zeros(T, 1); gross = zeros(T, 1);
pnl(3:end) = sum(w(1:end-2, :) .* R(3:end, :), 2);
gross(3:end) = sum(abs(w(1:end-2, :)), 2);
ok = keep & gross > 0;
x = pnl(ok);
s = mean(x) / std(x) * sqrt(252);
end

function E = ema_nan(X, span)
a = 2 / (span + 1);
E = nan(size(X));
prev = nan(1, size(X, 2));
for t = 1:size(X, 1)
    x = X(t, :);
    cur = prev;
    seed = isnan(prev) & ~isnan(x);
    upd = ~isnan(prev) & ~isnan(x);
    cur(seed) = x(seed);
    cur(upd) = a * x(upd) + (1 - a) * prev(upd);
    E(t, :) = cur;
    prev = cur;
end
end
