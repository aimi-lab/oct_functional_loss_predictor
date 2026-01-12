from torchmetrics import Metric, MetricCollection

def unpack_metrics(metrics: dict[str, Metric]) -> dict[str, Metric]:
    """
    Unpack the metrics to make sure they can be logged. 
    log_dict can only handle a dict with scalar values, thus we have to unpack tensors.
    """
    unpacked_metrics = {}
    for key, metric in metrics.items():
        if metric.dim() > 0:
            unpacked_metrics.update(expand_dict(key, metric))
        else:
            unpacked_metrics[key] = metric
    return unpacked_metrics

def expand_dict(key: str, values):
    assert isinstance(key, str), "Key must be a string"
    new_dict = {}
    for i, v in enumerate(values):
        new_dict[key + f"_{i}"] = v
    return new_dict
