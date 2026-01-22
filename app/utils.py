from functools import wraps
from typing import Any, Callable, List, Optional
import logging
import plotly.graph_objects as go
from app.data.processing import get_dataset

log = logging.getLogger(__name__)


def get_df_or_none(df_meta: Optional[dict]):
    """Return the cached DataFrame or ``None``."""
    if not df_meta:
        return None
    ds_id = df_meta.get("dataset_id")
    return get_dataset(ds_id) if ds_id else None


def empty_fig(msg="No data"):
    """Create a minimal Plotly figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def ensure_list(x: Any) -> List:
    """Coerce *x* to a list – ``None`` → ``[]``, ``str`` → ``[str]``."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]


def ensure_df(empty_return=None):
    """
    Decorator for callbacks that receive ``df_meta``.
    The wrapped function receives the actual DataFrame; if missing,
    ``empty_return`` is returned immediately.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(df_meta, *args, **kwargs):
            df = get_df_or_none(df_meta)
            if df is None:
                return empty_return
            return func(df, *args, **kwargs)

        return wrapper

    return decorator
