function beamform_subject(hm_path, sm_path, run_paths_csv, out_path, ft_path, qc_flag)
% BEAMFORM_SUBJECT  FieldTrip singleshell LCMV -> per-source PSD + intrinsic timescale.
%
% SPEC_hcp_meg_dynamics_v2 Stage 1a/2. For one HCP-MEG subject:
%   * ft_prepare_leadfield on the singleshell headmodel + 8004-source 2D sourcemodel
%   * LCMV beamformer (fixedori max-power, unit-noise-gain, 5% loading) per resting run
%   * broadband source time series -> Welch PSD (2 s Hann, 50%% overlap, 1-100 Hz)
%   * intrinsic timescale (INT) from the broadband source ACF: 3 definitions
% Saves per-run PSD + INT maps (v7 .mat) for the Python phase. Chunked over sources
% to bound memory. Units are harmonised by FieldTrip from each struct's .unit field.
%
% Args (all strings): hm_path sm_path run_paths_csv(';'-sep) out_path ft_path qc_flag('1'/'0')

addpath(ft_path); ft_defaults;
qc = strcmp(qc_flag, '1');
S = load(hm_path); headmodel = S.headmodel;
S = load(sm_path); sourcemodel = S.sourcemodel2d;
if ~isfield(sourcemodel, 'inside')
    sourcemodel.inside = true(size(sourcemodel.pos, 1), 1);
end
run_paths = strsplit(run_paths_csv, ';');
run_paths = run_paths(~cellfun(@isempty, run_paths));
nrun = numel(run_paths);
nsrc = size(sourcemodel.pos, 1);
brainstructure = sourcemodel.brainstructure(:);

CHUNK = 1000;
psd_runs = []; freqs = [];
int_runs = nan(nsrc, 3, nrun);          % [lag1e, area_to_zero, exp_tau]
qc_ts = [];

for r = 1:nrun
    L = load(run_paths{r}); data = L.data;
    fs = data.fsample;

    cfg = []; cfg.grad = data.grad; cfg.headmodel = headmodel;
    cfg.sourcemodel = sourcemodel; cfg.channel = data.label;
    cfg.normalize = 'no';
    lf = ft_prepare_leadfield(cfg, data);

    cfg = []; cfg.covariance = 'yes'; cfg.covariancewindow = 'all'; cfg.keeptrials = 'no';
    tl = ft_timelockanalysis(cfg, data);

    cfg = []; cfg.method = 'lcmv'; cfg.sourcemodel = lf; cfg.headmodel = headmodel;
    cfg.lcmv.keepfilter = 'yes'; cfg.lcmv.fixedori = 'yes';
    cfg.lcmv.weightnorm = 'unitnoisegain'; cfg.lcmv.lambda = '5%';
    src = ft_sourceanalysis(cfg, tl);

    inside = find(src.inside(:));
    filters = src.avg.filter;
    alldata = cat(2, data.trial{:});                 % nchan x ntime (broadband)
    ntime = size(alldata, 2);

    win = round(2 * fs); nov = round(win / 2);
    if isempty(freqs)
        [~, fall] = pwelch(alldata(1, :)', hann(win), nov, [], fs);
        fmask = (fall >= 1) & (fall <= 100);
        freqs = fall(fmask);
        psd_runs = nan(nsrc, numel(freqs), nrun);
    end

    for c = 1:CHUNK:numel(inside)
        idx = inside(c:min(c + CHUNK - 1, numel(inside)));
        W = zeros(numel(idx), size(alldata, 1));
        for i = 1:numel(idx)
            W(i, :) = filters{idx(i)};
        end
        sts = W * alldata;                            % nchunk x ntime
        [pxx, ~] = pwelch(sts', hann(win), nov, [], fs);   % nfreq_all x nchunk
        psd_runs(idx, :, r) = pxx(fmask, :)';
        [i1e, iar, itau] = local_int(sts, fs);
        int_runs(idx, 1, r) = i1e;
        int_runs(idx, 2, r) = iar;
        int_runs(idx, 3, r) = itau;
    end

    if qc && r == 1
        step = max(1, floor(ntime / 20000));          % ~<=20k samples, downsampled
        qc_ts = single(sts(1:min(200, size(sts,1)), 1:step:end));  % first 200 src only
    end
    clear alldata sts lf src tl data;
end

meta = struct('fs', fs, 'nrun', nrun, 'nsrc', nsrc, 'win_s', 2, 'overlap', 0.5, ...
              'weightnorm', 'unitnoisegain', 'lambda', '5%', 'ft', ft_path);
save(out_path, 'psd_runs', 'int_runs', 'freqs', 'brainstructure', 'meta', 'qc_ts', '-v7');
fprintf('BEAMFORM_DONE %s nrun=%d nsrc=%d nfreq=%d\n', out_path, nrun, nsrc, numel(freqs));
end


function [lag1e, area0, tau] = local_int(sts, fs)
% Per-source intrinsic timescale from the broadband ACF.
%  lag1e : lag (s) at which the normalised ACF first drops to 1/e (linear interp)
%  area0 : area (s) under the ACF up to the first zero crossing (trapz)
%  tau   : exponential-decay tau (s) from an LS fit of log(ACF) over positive lags to 1/e
n = size(sts, 1);
lag1e = nan(n, 1); area0 = nan(n, 1); tau = nan(n, 1);
maxlag = round(1.0 * fs);                              % up to 1 s of lags
thr = exp(-1);
dt = 1 / fs;
for i = 1:n
    x = sts(i, :) - mean(sts(i, :));
    if all(x == 0), continue; end
    ac = xcorr(x, maxlag, 'coeff');
    ac = ac(maxlag + 1:end);                           % lags 0..maxlag
    % lag to 1/e (first crossing, linear interp)
    below = find(ac < thr, 1, 'first');
    if ~isempty(below) && below > 1
        a = ac(below - 1); b = ac(below);
        frac = (a - thr) / (a - b);
        lag1e(i) = (below - 2 + frac) * dt;
    end
    % area to first zero crossing
    zc = find(ac <= 0, 1, 'first');
    if isempty(zc), zc = numel(ac); end
    area0(i) = trapz(ac(1:zc)) * dt;
    % exponential-decay tau over lags where ac still > 1/e
    k = find(ac <= thr, 1, 'first'); if isempty(k), k = numel(ac); end
    if k >= 3
        lags = (0:k-1)' * dt; y = ac(1:k)';
        good = y > 0;
        if sum(good) >= 3
            p = polyfit(lags(good), log(y(good)), 1);
            if p(1) < 0, tau(i) = -1 / p(1); end
        end
    end
end
end
