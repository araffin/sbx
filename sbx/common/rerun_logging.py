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
    time_range = rr.TimeRange(
        start=rrb.TimeRangeBoundary.cursor_relative(seq=-100),
        end=rrb.TimeRangeBoundary.cursor_relative(seq=100),
    )
    time_range_qf = rr.TimeRange(
        start=rrb.TimeRangeBoundary.cursor_relative(seq=-10),
        end=rrb.TimeRangeBoundary.cursor_relative(seq=10),
    )
    times_series = [
        rrb.TimeSeriesView(
            name=f"Action {i + 1}",
            origin=f"/actions_{i + 1}",
            axis_x=rrb.TimeAxis(
                view_range=time_range,
                zoom_lock=True,
            ),
            axis_y=rrb.ScalarAxis(range=(-1, 1), zoom_lock=True),
            plot_legend=rrb.PlotLegend(visible=False),
        )
        for i in range(n_actions)
    ]
    times_series.extend(
        [
            rrb.TimeSeriesView(
                name="Q values",
                origin="/qf_values",
                axis_x=rrb.TimeAxis(
                    view_range=time_range_qf,
                    zoom_lock=True,
                ),
                plot_legend=rrb.PlotLegend(visible=True),
            ),
            rrb.TimeSeriesView(
                name="Min Q values",
                origin="/min_qf_values",
                axis_x=rrb.TimeAxis(
                    view_range=time_range_qf,
                    zoom_lock=True,
                ),
                plot_legend=rrb.PlotLegend(visible=True),
            ),
        ]
    )
    blueprint = rrb.Grid(*times_series)
    rr.send_blueprint(blueprint, make_active=True)

    for i in range(n_actions):
        rr.log(f"actions_{i + 1}/pi", rr.SeriesLines(colors=[31, 119, 180], names="pi(s)", widths=2), static=True)
        rr.log(f"actions_{i + 1}/cem", rr.SeriesLines(colors=[255, 127, 14], names="cem(s)", widths=2), static=True)

    rr.log("min_qf_values/pi", rr.SeriesLines(widths=2), static=True)
    rr.log("min_qf_values/cem", rr.SeriesLines(widths=2), static=True)

    for q_value in ["qf1", "qf2"]:
        rr.log(f"qf_values/{q_value}/pi", rr.SeriesLines(widths=2), static=True)
        rr.log(f"qf_values/{q_value}/cem", rr.SeriesLines(widths=2), static=True)


def log_step(step: int, actions: np.ndarray, cem_actions: np.ndarray, qf_values: np.ndarray | None = None) -> None:
    import rerun as rr

    rr.set_time("step", sequence=step)

    for i in range(actions.shape[0]):
        rr.log(f"actions_{i + 1}/pi", rr.Scalars(actions[i]))
        rr.log(f"actions_{i + 1}/cem", rr.Scalars(cem_actions[i]))

    if qf_values is not None:
        for idx, qf in enumerate(["qf1", "qf2"]):
            rr.log(f"qf_values/{qf}/pi", rr.Scalars(qf_values[idx, 0].item()))
            rr.log(f"qf_values/{qf}/cem", rr.Scalars(qf_values[idx, 1].item()))
        rr.log("min_qf_values/pi", rr.Scalars(np.min(qf_values[:, 0])))
        rr.log("min_qf_values/cem", rr.Scalars(np.min(qf_values[:, 1])))
