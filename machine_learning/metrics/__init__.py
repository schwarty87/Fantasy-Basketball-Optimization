# This file ensures that all metric modules are imported when the 'metrics' package is accessed, which triggers the
# @register_metric decorator to populate the METRIC_REGISTRY.

from . import metrics