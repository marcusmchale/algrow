"""Read in a sample details file and perform lsgr analysis"""
import logging
from pandas import read_csv, to_datetime, merge, Timedelta, Series
import numpy as np
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pathlib import Path

logger = logging.getLogger(__name__)


def analyse(args):
    area_in = args.area_file
    area_header = ['File', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area', 'RGB', "Lab"]
    area_analyser = AreaAnalyser(area_in, args.samples, args, area_header)
    area_analyser.fit_all(args.fit_start, args.fit_end)
    area_analyser.write_results(args.out_dir, group_plots=True)


class AreaAnalyser:
    def __init__(self, area_csv, id_csv, args, area_header):
        self.args = args
        self.area_header = area_header
        self.logger = logging.getLogger(__name__)
        columns = ["Block", "Unit", "Time", "Area", "Group"]
        id_df = self._load_id(id_csv)
        area_df = self._load_area(area_csv)
        self.df = merge(area_df, id_df, on=["Block", "Unit"])[columns]
        if self.df.shape[0] == 0:
            raise ValueError(
                "No data found applying the map to the area file.\n "
                "Please check the values for 'Block' and 'Unit' correspond across these files."
            )
        self.df = self.df.set_index(["Block", "Unit", "Time"]).sort_index()

    def _load_area(self, area_csv):
        self.logger.debug(f"Load area data from file: {area_csv}")
        with open(area_csv) as area_csv:
            area_df = read_csv(
                area_csv,
                sep=",",
                names=self.area_header,
                header=0,
                dtype={
                    "File": str,
                    'Block': str,
                    'Plate': str,
                    "Unit": int,
                    'Time': str,
                    'Pixels': int,
                    'Area': np.float64,
                    'RGB': str,
                    'Lab': str
                }
            )
            area_df["Time"] = to_datetime(area_df["Time"])

        return area_df

    def _load_id(self, id_csv):
        self.logger.debug(f"Load sample identities from file: {id_csv}")
        with open(id_csv) as id_csv:
            id_df = read_csv(
                id_csv,
                sep=",",
                names=["Block", "Unit", "Group"],
                header=0,
                dtype={"Block": str, "Unit": int, 'Area': np.float64}
            )
        return id_df

    def _fit(self, group):
        self.logger.debug("Fit group")
        group = group[group.log_area != -np.inf]

        if group.elapsed_m.unique().size <= 1:
            return np.nan, np.nan, np.nan
        polynomial, svd = np.polynomial.Polynomial.fit(group.elapsed_m, group.log_area, deg=1, full=True)
        coef = polynomial.convert().coef
        intercept = coef[0]
        slope = coef[1]
        try:
            rss = svd[0][0]
        except IndexError:
            return intercept, slope, np.nan
        return intercept, slope, rss

    def fit_all(self, fit_start, fit_end):
        self.logger.debug(f"Perform fit on log transformed values from day {fit_start} to day {fit_end}")
        start = self.df.index.get_level_values('Time').min()
        day0_start = start.replace(hour=0, minute=0)
        df = self.df.copy()
        df["elapsed_D"] = (df.index.get_level_values("Time") - day0_start) // Timedelta(days=1)
        df = df[(df.elapsed_D >= fit_start) & (df.elapsed_D <= fit_end)]
        df["elapsed_m"] = (df.index.get_level_values("Time") - start) // Timedelta(minutes=1)
        with np.errstate(divide='ignore'):
            df["log_area"] = np.log(df["Area"])  # 0 values are -Inf which are then ignored in the fit

        df[["Intercept", "Slope", "RSS"]] = df.groupby(["Block", "Unit"], group_keys=True, dropna=False).apply(self._fit).apply(Series)
        # RGR calculation:
        # slope of the fit is with minutes on the x-axis so convert to days (1440 minutes/day)
        # then express as percent and
        # round to 2 significant figures
        df["RGR"] = round(df["Slope"] * 1440 * 100, 2)
        self.df = self.df.join(df[["Slope", "Intercept", "RGR", "RSS", "elapsed_m"]])

    def _get_df_mask(self, blocks: list = None, units: list = None, groups: list = None):
        self.logger.debug("get mask")
        mask = np.full(self.df.shape[0], True)
        if blocks and any(blocks):
            mask = mask & np.isin(self.df.Block, blocks)
        if blocks and any(units):
            mask = mask & np.isin(self.df.Unit, units)
        if groups and any(groups):
            mask = mask & np.isin(self.df.Group, groups)
        return mask

    def draw_plot(
            self,
            ax,
            groups=None,
            plot_fit=None,
            outliers=None
    ):
        self.logger.debug("draw plot")
        mask = self._get_df_mask(groups=groups)
        df = self.df[mask]
        df = df.groupby(["Block", "Unit"], group_keys=True, dropna=False)
        markers = list(Line2D.markers.keys())
        for i, (name, group) in enumerate(df):
            ax.scatter(
                group.index.get_level_values("Time"),
                group["Area"],
                s=30,
                marker=markers[i],
                label=f"Block {name[0]}, Unit {name[1]}. RGR: {group.RGR.dropna().unique()}"
            )
            if plot_fit:
                # todo here we are converting to string to handle matches to np.nan, find a better way
                if str(name) in [str(i) for i in plot_fit]:
                    # todo here we are converting to string to handle matches to np.nan, find a better way
                    ax.plot(
                        group.index.get_level_values("Time"),
                        np.exp(group.Slope * group.elapsed_m + group.Intercept),
                        linestyle="dashed" if outliers and str(name) in [str(i) for i in outliers] else "solid"
                    )
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.set_ylabel("Area (mm²)")
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', labelrotation=45)
        return ax

    def summarise(self):
        summary = self.df[["Group", "RGR", "RSS"]].droplevel("Time").drop_duplicates()  # .dropna()
        if summary.size != 0:
            d = np.sqrt(summary.RSS)
            # identify atypical models from RSS
            q75, q25 = np.percentile(d, [75, 25])
            iqr = q75 - q25
            median = d.median()
            summary["ModelFitOutlier"] = d > median + (1.5 * iqr)
        return summary

    def write_results(self, outdir, rgr_plot=True, group_plots=False):
        self.logger.debug("Write out results")
        summary = self.summarise()
        if summary.size == 0:
            self.logger.debug("No results to write")
            return
        rgr_out = Path(outdir, "RGR.csv")
        rgr_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(rgr_out)
        mean_rgr = summary[~summary.ModelFitOutlier].groupby("Group", dropna=False).mean()
        mean_rgr_out = Path(outdir, "RGR_mean.csv")
        mean_rgr["RGR"].to_csv(mean_rgr_out)
        if rgr_plot:
            figsize = ((len(set(self.df.Group)) / 10 + 5), 5)  # tuple of (width, heighqt) in inches
            fig = Figure()
            ax = fig.add_subplot()
            summary.boxplot("RGR", by="Group", rot=90, figsize=figsize, ax=ax)
            ax.set_ylabel("RGR (%/day")
            plot_path = Path(outdir, "Figures", "RGR.png")
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(plot_path), dpi=300)
            # similar plot with model fit outliers removed
            fig = Figure()
            ax = fig.add_subplot()
            sub_summary = summary[~summary.ModelFitOutlier]
            sub_summary.boxplot("RGR", by="Group", rot=90, figsize=figsize, ax=ax)
            ax.set_ylabel("RGR (%/day")
            plot_path = Path(outdir, "Figures", "RGR_ModelFitOutliers_removed.png")
            fig.savefig(str(plot_path), dpi=300)
        if group_plots:
            group_plot_dir = Path(outdir, "Figures", "group_plots")
            group_plot_dir.mkdir(parents=True, exist_ok=True)
            for group in set(self.df.Group):
                fig = Figure()
                ax: Axes = fig.add_subplot()
                to_plot_fit = summary[(summary.Group == group)].index.to_list()
                outliers = summary[summary.ModelFitOutlier].index.to_list()
                self.draw_plot(ax, plot_fit=to_plot_fit, groups=[group], outliers=outliers)
                plot_path = Path(group_plot_dir, f"{group}.png")
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(plot_path), bbox_inches='tight')


