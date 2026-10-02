"""Macro F0.5 exactly as the challenge defines it (per Source-1 entity, singletons included)."""
import polars as pl


def macro_f05(pred, truth, s1_ids):
    """pred, truth: frames (rid1, rid2). s1_ids: Series of evaluated rid1."""
    base = pl.DataFrame({'rid1': s1_ids})
    tp = pred.join(truth, on=['rid1', 'rid2']).group_by('rid1').len().rename({'len': 'tp'})
    npred = pred.group_by('rid1').len().rename({'len': 'np'})
    ntrue = truth.group_by('rid1').len().rename({'len': 'nt'})
    d = (base.join(tp, on='rid1', how='left').join(npred, on='rid1', how='left')
             .join(ntrue, on='rid1', how='left').fill_null(0))
    p = pl.col('tp') / pl.col('np')
    r = pl.col('tp') / pl.col('nt')
    f = (pl.when((pl.col('nt') == 0) & (pl.col('np') == 0)).then(1.0)
           .when(pl.col('tp') == 0).then(0.0)
           .otherwise(1.25 * p * r / (0.25 * p + r)))
    d = d.with_columns(f.alias('f'))
    return d['f'].mean(), d
