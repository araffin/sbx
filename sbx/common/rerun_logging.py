import os

import numpy as np

# Trick from Flo: accepts Yes, YES, True, true, 1
ENABLE_RERUN = os.environ.get("ENABLE_RERUN", "0")[:1].lower() in ("1", "y", "t")


def init_rerun(
    env_name: str = "debug",
    rerun_url: str = "rerun+http://192.168.178.57:9876/proxy",
    n_actions: int = 1,
) -> None:
    if not ENABLE_RERUN:
        return
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init(f"rl_debug_cem_{env_name}")
    rr.connect_grpc(rerun_url)

    times_series = [
        rrb.TimeSeriesView(
            name=f"Action {i + 1}",
            origin=f"/actions_{i + 1}",
            axis_x=rrb.TimeAxis(
                view_range=rr.TimeRange(
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-100),
                    end=rrb.TimeRangeBoundary.cursor_relative(seq=100),
                ),
                zoom_lock=True,
            ),
            axis_y=rrb.ScalarAxis(range=(-1, 1), zoom_lock=True),
            plot_legend=rrb.PlotLegend(visible=False),
            time_ranges=[
                rrb.VisibleTimeRange(
                    "time",
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-100),
                    end=rrb.TimeRangeBoundary.cursor_relative(seq=100),
                ),
            ],
        )
        for i in range(n_actions)
    ]
    blueprint = rrb.Grid(*times_series)
    rr.send_blueprint(blueprint, make_active=True)

    for i in range(n_actions):
        rr.log(f"actions_{i + 1}/pi", rr.SeriesLines(colors=[31, 119, 180], names="pi(s)", widths=2), static=True)
        rr.log(f"actions_{i + 1}/cem", rr.SeriesLines(colors=[255, 127, 14], names="cem(s)", widths=2), static=True)


def log_step(step: int, actions: np.ndarray, cem_actions: np.ndarray) -> None:
    import rerun as rr

    rr.set_time("step", sequence=step)

    for i in range(actions.shape[0]):
        rr.log(f"actions_{i + 1}/pi", rr.Scalars(actions[i]))
        rr.log(f"actions_{i + 1}/cem", rr.Scalars(cem_actions[i]))
