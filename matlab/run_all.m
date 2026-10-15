function run_all()
%RUN_ALL Phase 7 entry point: matlab -batch "cd('<alphalab>/matlab'); run_all"
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
studyDir = fullfile(root, 'reports', 'study');
exportDir = fullfile(studyDir, 'export');
outDir = fullfile(root, 'reports', 'matlab');
if ~exist(outDir, 'dir'); mkdir(outDir); end
v = ver;
lic = {'Statistics_Toolbox', 'Distrib_Computing_Toolbox', 'Financial_Toolbox'};
fid = fopen(fullfile(outDir, 'environment.txt'), 'w');
fprintf(fid, 'MATLAB %s\n', version);
for i = 1:numel(lic); fprintf(fid, 'license %s: %d\n', lic{i}, license('test', lic{i})); end
for i = 1:numel(v); fprintf(fid, '%s %s\n', v(i).Name, v(i).Version); end
fclose(fid);
t0 = tic;
crosscheck_metrics(exportDir, outDir);   fprintf('crosscheck done\n');
bootstrap_sharpe(exportDir, outDir, 2000); fprintf('bootstrap done\n');
regime_analysis(exportDir, outDir);      fprintf('regimes done\n');
param_surface(studyDir, outDir);         fprintf('surface done\n');
parfor_sweep(exportDir, outDir);         fprintf('parfor done\n');
fprintf('total %.1f s\n', toc(t0));
end
