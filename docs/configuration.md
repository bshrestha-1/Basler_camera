# Configuration

GLAS is configured through a single YAML file, merged over built-in
defaults and validated with [Pydantic v2](https://docs.pydantic.dev/) before use.

## File resolution order

`glas.settings.Settings.load()` looks for a configuration file in this
order, using the first one found:

1. An explicit path passed by the caller (e.g. `Settings.load(config_path=...)`,
   or `glas config show --path <file>` on the CLI).
2. The path in the `GLAS_CONFIG` environment variable, if set.
3. `./glas.yaml` (current working directory).
4. `~/.config/glas/config.yaml`.

If none of these exist, GLAS runs entirely on built-in defaults — no
configuration file is required to get started.

## Schema

A configuration file only needs to specify the keys it wants to override;
anything omitted falls back to the default. The full schema, after
defaults are merged in, is:

```yaml
paths:
  data_dir: ~/glas_data       # root directory for recorded experiments
  log_dir: ~/glas_data/logs   # directory for log files

logging:
  level: INFO                 # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: glas.log              # log file name within log_dir
  max_bytes: 10485760         # rotate after this many bytes (10 MiB)
  backup_count: 5             # number of rotated log files to keep
  console: true                # also log to stderr
```

Validation is performed by a Pydantic model (`glas.settings.Settings`, built
from private nested `_PathsConfig`/`_LoggingConfig` models mirroring the file
structure above). Invalid files raise `glas.exceptions.JSONValidationError`
(`Settings.from_dict()` translates Pydantic's `ValidationError` into it via
`JSONValidationError.from_pydantic()`), which lists every violation found,
not just the first one.

## CLI commands

```bash
# Write a default configuration file (won't overwrite an existing file
# unless --force is given)
glas config init --path ~/.config/glas/config.yaml

# Validate a configuration file against the schema
glas config validate ~/.config/glas/config.yaml

# Load configuration (defaults + file, if found) and print the resolved
# settings
glas config show
```

## Programmatic use

```python
from glas.settings import Settings

settings = Settings.load()          # follows the resolution order above
settings.ensure_directories()       # create data_dir / log_dir if missing

print(settings.data_dir)
print(settings.log_level)
```

## Adding new configuration keys (for future phases)

Later phases (camera settings, acquisition tuning, etc.) will extend
`DEFAULT_CONFIG`, `_PathsConfig`/`_LoggingConfig` (or a new nested model),
and `Settings` itself in `glas/settings.py`. `glas.config` itself stays
domain-agnostic and validation-free (`read_yaml_file`, `deep_merge`,
`find_config_file`, `load_config`) -- it does not need to change to support
new settings; only the Pydantic models that interpret its output do.
