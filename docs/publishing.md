# Publishable Results: Styling, Statistics, Comparison, and Reports

Phase 20 adds the layer between "an analysis ran" and "a figure ready for
a paper": consistent journal-quality plot styling across every existing
`plot_*` function, proper statistics (confidence intervals via Student's
t, not a fixed z-score; linear regression) for repeated-trial data,
generic multi-run parameter sweeps built on the existing experiment
manager, and a self-contained HTML report per recording.

## Journal-quality plot styling (`glas.plotting`)

Every `plot_*` function across `glas.analysis.brazil_nut`,
`glas.analysis.convection`, `glas.analysis.packing`,
`glas.analysis.segregation`, and `glas.accelerometer` calls
`glas.plotting.apply_publication_style()` before drawing and
`glas.plotting.savefig_publication()` to save -- a consistent,
colorblind-safe (Okabe-Ito) palette, readable font sizes, a light grid,
and 300 DPI raster output, applied uniformly rather than each function
picking up matplotlib's mismatched defaults independently.

```python
from glas.plotting import apply_publication_style, style_axes, savefig_publication
import matplotlib.pyplot as plt

apply_publication_style()
fig, ax = plt.subplots()
ax.plot(x, y)
style_axes(ax)  # removes the top/right spines
savefig_publication(fig, Path("figure.pdf"))  # or .png, .svg -- format from extension
```

Nothing about *what* gets plotted changes -- only consistent styling and
resolution, and a vector format (`.pdf`/`.svg`) works exactly as before,
since matplotlib infers the format from the file extension.

## Statistics (`glas.stats`)

`describe()` computes sample mean, standard deviation, standard error,
and a confidence interval via Student's t distribution (not a fixed
z-score, which matters for the small repeated-trial counts typical in a
granular-physics lab). `linear_fit()` wraps ordinary least-squares
regression (slope, intercept, standard errors, R², p-value).

```python
from glas.stats import describe, linear_fit

stats = describe([4.8, 5.1, 5.3, 4.9], confidence_level=0.95)
print(stats.mean, stats.sem, stats.ci_low, stats.ci_high)

fit = linear_fit(gammas, rise_times)
print(f"slope={fit.slope:.3f} R²={fit.r_squared:.3f} p={fit.p_value:.4g}")
```

Built on `scipy.stats` rather than hand-rolled formulas -- unlike
particle linking (see `glas.analysis.tracking_utils`), there's no simpler
correct alternative to the t-distribution and least-squares regression
that an established statistics library doesn't already implement
carefully. `scipy` is a core GLAS dependency starting with this release.

## Multi-run comparison and parameter sweeps (`glas.analysis.comparison`)

A single recording's analysis gives one point estimate. A publishable
result almost always needs the next step: how did this measurement
change across many recordings at different Gamma / fill depth / grain
size -- a parameter sweep, with repeated trials at each condition
averaged and given real uncertainty.

`compare_runs()` is deliberately generic rather than hardcoded to one
analysis: it takes a *parameter extractor* (how to read the independent
variable, typically a `glas.experiment.PhysicalParameters` field) and a
*metric extractor* (how to compute the dependent variable -- any
existing `analyze_*` function's output).

```python
from glas.experiment import ExperimentManager, get_physical_parameters
from glas.analysis import analyze_brazil_nut
from glas.analysis.comparison import compare_runs, plot_parameter_sweep, export_sweep_csv

manager = ExperimentManager(Path("~/glas_data").expanduser())
summaries = manager.search_experiments(tag="brazil-nut")

result = compare_runs(
    summaries,
    parameter_fn=lambda md: get_physical_parameters(md).target_acceleration_g,
    metric_fn=lambda folder: analyze_brazil_nut(folder).rise_time_s,
    parameter_name="Gamma",
    metric_name="Rise time (s)",
)
plot_parameter_sweep(result, Path("rise_time_vs_gamma.pdf"))
export_sweep_csv(result, Path("rise_time_vs_gamma.csv"))
```

A recording whose metric extraction fails (too few particles, too few
frames) is skipped with a logged warning rather than aborting the whole
sweep -- a comparison across dozens of recordings shouldn't fail outright
because one of them is unusable. `result.fit` is a `LinearFitResult`
computed across group means once at least 3 distinct parameter values are
found.

### CLI

```bash
glas compare ~/glas_data --parameter target-acceleration-g --metric brazil-nut-rise-time \
    --tag brazil-nut --plot sweep.png --csv sweep.csv
```

`--parameter` accepts `target-acceleration-g`, `frequency-hz`,
`amplitude-mm`, `fill-depth-mm`, `grain-diameter-mm`; `--metric` accepts
`brazil-nut-rise-time`, `brazil-nut-mean-velocity`, `packing-fraction`,
`segregation-index`, `convection-circulation`. From Python, `compare_runs()`
accepts *any* extractor function, not just these five -- the CLI's list
is a convenience mapping over the most common ones.

## Automated experiment reports (`glas.report`)

`generate_report()` runs every analysis GLAS has for one recording
(tracking, Brazil nut, convection, packing, segregation, and optionally
vibration if an accelerometer CSV is given) and writes a single,
self-contained HTML file -- summary statistics and a publication-styled
plot per section, base64-embedded directly into the HTML, so there's
nothing to lose track of.

```python
from glas.report import generate_report

generate_report(
    Path("~/glas_data/Run0001").expanduser(),
    Path("report.html"),
    accelerometer_csv=Path("shaker_run.csv"),  # optional
)
```

```bash
glas report ~/glas_data/Run0001 report.html --accelerometer-csv shaker_run.csv
```

An individual analysis failing partway through (too few particles for
Brazil nut, too few frames for convection) doesn't abort the report --
that section is shown as "Skipped: \<reason\>" instead, since the operator
still wants the report for everything that *did* work. Only a total
failure (the dataset's own frames can't be read at all) raises
`glas.exceptions.ReportError`. The GUI's analysis panel has a matching
**Report** tab.
