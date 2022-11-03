"""Read in a sample details file and perform lsgr analysis"""
import logging
from pandas import read_csv, to_datetime, merge  # todo consider replacing the existing use of csv reader/writer with pandas?
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class AreaAnalyser:
    def __init__(self, area_csv, id_csv, area_header):
        self.area_header = area_header
        self.logger = logging.getLogger(__name__)
        columns = ["Block", "Unit", "Time", "Area", "Group"]
        self.df = merge(
            self._load_area(area_csv), self._load_id(id_csv), on=["Block", "Unit"]
        )[columns].set_index(["Block", "Unit", "Time"]).sort_index()

    def _load_area(self, area_csv):
        self.logger.debug(f"Load area data from file: {area_csv}")
        with open(area_csv) as area_csv:
            area_df = read_csv(
                area_csv,
                sep=",",
                names=self.area_header,
                header=0,
                usecols=["ImageFile", "Unit", "Area"],
                dtype={"ImageFile": str, "Unit": int, 'Area': np.float64}
            )
            area_df["Time"] = to_datetime(area_df["ImageFile"].str.extract(args.time_regex))
            area_df["Block"] = area_df["ImageFile"].str.extract(args.block_regex).astype(int)
        return area_df

    def _load_id(self, id_csv):
        self.logger.debug(f"Load sample identities from file: {id_csv}")
        with open(id_csv) as id_csv:
            id_df = read_csv(
                id_csv,
                sep=",",
                names=["Block", "Unit", "Group"],
                header=0,
                dtype={"Block": int, "Unit": int, 'Area': np.float64}
            )
        return id_df

    def _fit(self, group):
        self.logger.debug("Fit group")
        group = group[group.log_area != -np.inf]
        if group.elapsed_m.unique().size == 1:
            return np.nan, np.nan, np.nan
        polynomial, svd = np.polynomial.Polynomial.fit(group.elapsed_m, group.log_area, deg=1, full=True)
        coef = polynomial.convert().coef
        intercept = coef[0]
        slope = coef[1]
        rss = svd[0][0]
        return intercept, slope, rss

    def fit_all(self, fit_start, fit_end):
        self.logger.debug(f"Perform fit on log transformed values from day {fit_start} to day {fit_end}")
        start = self.df.index.get_level_values('Time').min()
        day0_start = start.replace(hour=0, minute=0)
        df = self.df.copy()
        df["elapsed_D"] = (df.index.get_level_values("Time") - day0_start).astype('timedelta64[D]')
        df = df[(df.elapsed_D >= fit_start) & (df.elapsed_D <= fit_end)]
        df["elapsed_m"] = (df.index.get_level_values("Time") - start).astype('timedelta64[m]')
        with np.errstate(divide='ignore'):
            df["log_area"] = np.log(df["Area"])  # 0 values are -Inf which are then ignored in the fit
        df["fit"] = df.groupby(["Block", "Unit"], group_keys=True).apply(self._fit)
        df[["intercept", "slope", "RSS"]] = df.fit.to_list()
        df["RGR"] = round(df["slope"] * 1440 * 100, 2)
        self.df = self.df.join(df[["slope", "intercept", "RGR", "RSS", "elapsed_m"]])

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
            plot_fit=None
    ):
        self.logger.debug("draw plot")
        mask = self._get_df_mask(groups=groups)
        df = self.df[mask]
        df = df.groupby(["Block", "Unit"], group_keys=True)
        for i, (name, group) in enumerate(df):
            ax.scatter(
                group.index.get_level_values("Time"),
                group["Area"],
                s=1,
                label=f"Block {name[0]}, Unit {name[1]}. RGR: {group.RGR.dropna().unique()}"
            )
            if plot_fit:
                if name in plot_fit:
                    ax.plot(
                        group.index.get_level_values("Time"),
                        np.exp(group.slope * group.elapsed_m + group.intercept)
                    )
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', labelrotation=45)
        return ax

    def summarise(self):
        summary = self.df[["Group", "RGR", "RSS"]].droplevel("Time").drop_duplicates().dropna()
        if summary.size != 0:
            # identify atypical models from RSS
            q75, q25 = np.percentile(summary.RSS, [75, 25])
            iqr = q75 - q25
            median = summary.RSS.median()
            summary["ModelFitOutlier"] = summary.RSS > median + (1.5 * iqr)
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
        mean_rgr = summary[~summary.ModelFitOutlier].groupby("Group").mean()
        mean_rgr_out = Path(outdir, "RGR_mean.csv")
        mean_rgr["RGR"].to_csv(mean_rgr_out)
        if rgr_plot:
            fig, ax = plt.subplots()
            summary.boxplot("RGR", by="Group", rot=90)
            plt.tight_layout()
            plt.savefig(Path(outdir, "RGR.png"))
            plt.close(fig)
            # similar plot with model fit outliers removed
            fig, ax = plt.subplots()
            sub_summary = summary[~summary.ModelFitOutlier]
            sub_summary.boxplot("RGR", by="Group", rot=90)
            plt.title("Grouped (low quality RGR models removed)")
            plt.tight_layout()
            plt.savefig(Path(outdir, "RGR_less_outliers.png"))
            plt.close(fig)
        if group_plots:
            group_plot_dir = Path(outdir, "group_plots")
            group_plot_dir.mkdir(parents=True, exist_ok=True)
            for group in set(self.df.Group):
                to_plot_fit = summary[(summary.Group == group) & ~summary.ModelFitOutlier].index.to_list()
                fig, ax = plt.subplots()
                self.draw_plot(ax, plot_fit=to_plot_fit, groups=[group])
                plt.tight_layout()
                plt.savefig(Path(group_plot_dir, f"{group}.png"))
                plt.close(fig)


