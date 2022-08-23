"""Read in a sample details file and perform lsgr analysis"""
import logging
from pandas import read_csv, to_datetime, merge  # todo consider replacing the existing use of csv reader/writer with pandas?
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import area_header, datetime_regex, tank_regex


class AreaResult:
    def __init__(self, area_csv, id_csv):
        self.logger = logging.getLogger(__name__)
        columns = ["tank", "well", "time", "mm²", "strain"]
        self.df = merge(
            self._load_area(area_csv), self._load_id(id_csv), on=["tank", "well"]
        )[columns].set_index(["tank", "well", "time"]).sort_index()

    def _load_area(self,area_csv):
        self.logger.debug(f"Load area data from file: {area_csv}")
        with open(area_csv) as area_csv:
            area_df = read_csv(
                area_csv,
                sep=",",
                names=area_header,
                header=0,
                usecols=["filename", "well", "mm²"],
                dtype={"filename": str, "well": int, 'mm²': np.float64}
            )
            area_df["time"] = to_datetime(area_df["filename"].str.extract(datetime_regex))
            area_df["tank"] = area_df["filename"].str.extract(tank_regex).astype(int)
        return area_df

    def _load_id(self, id_csv):
        self.logger.debug(f"Load sample identities from file: {id_csv}")
        with open(id_csv) as id_csv:
            id_df = read_csv(
                id_csv,
                sep=",",
                names=["tank", "well", "strain"],
                header=0,
                dtype={"tank": int, "well": int, 'mm²': np.float64}
            )
        return id_df

    def _fit(self, group):
        self.logger.debug("Fit group")
        group = group[group.log_area != -np.inf]
        return np.polyfit(group.elapsed_m, group.log_area, deg=1, cov=False)

    def fit_all(self, fit_start, fit_end):
        self.logger.debug(f"Perform fit on log transformed values from day {fit_start} to day {fit_end}")
        start = self.df.index.get_level_values('time').min()
        day0_start = start.replace(hour=0, minute=0)
        df = self.df.copy()
        df["elapsed_D"] = (df.index.get_level_values("time") - day0_start).astype('timedelta64[D]')
        df = df[(df.elapsed_D >= fit_start) & (df.elapsed_D <= fit_end)]
        df["elapsed_m"] = (df.index.get_level_values("time") - start).astype('timedelta64[m]')
        with np.errstate(divide='ignore'):
            df["log_area"] = np.log(df["mm²"])  # 0 values are -Inf which are then ignored in the fit
        df["fit"] = df.groupby(["tank", "well"]).apply(self._fit)
        df[["slope", "intercept"]] = df.fit.to_list()
        df["RGR"] = round(df["slope"] * 1440 * 100, 2)
        self.df = self.df.join(df[["slope", "intercept", "RGR", "elapsed_m"]])

    def _get_df_mask(self, tanks: list = None, wells: list = None, strains: list = None):
        self.logger.debug("get mask")
        mask = np.full(self.df.shape[0], True)
        if tanks and any(tanks):
            mask = mask & np.isin(self.df.tank, tanks)
        if tanks and any(wells):
            mask = mask & np.isin(self.df.well, wells)
        if strains and any(strains):
            mask = mask & np.isin(self.df.strain, strains)
        return mask

    def draw_plot(
            self,
            ax,
            strains=None,
            fit=False
    ):
        self.logger.debug("draw plot")
        mask = self._get_df_mask(strains=strains)
        df = self.df[mask]
        df = df.groupby(["tank", "well"])
        for i, (name, group) in enumerate(df):
            ax.scatter(
                group.index.get_level_values("time"),
                group["mm²"],
                s=1,
                label=f"Tank {name[0]}, Well {name[1]}. RGR: {group.RGR.dropna().unique()}"
            )
            if fit:
                ax.plot(
                    group.index.get_level_values("time"),
                    np.exp(group.slope * group.elapsed_m + group.intercept)
                )
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', labelrotation=45)
        return ax

    def write_results(self, outdir, rgr_plot=True, strain_plots=False):
        self.logger.debug("write out results")
        summary = self.df[["strain", "RGR"]].droplevel("time").drop_duplicates().dropna()
        rgr_out = Path(outdir, "RGR.csv")
        rgr_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(rgr_out)
        if rgr_plot:
            fig, ax = plt.subplots()
            summary.boxplot(by="strain", rot=90)
            plt.tight_layout()
            plt.savefig(Path(outdir, "RGR.png"))
            plt.close(fig)
        if strain_plots:
            for strain in set(self.df.strain):
                fig, ax = plt.subplots()
                self.draw_plot(ax, fit=True, strains=[strain])
                plt.tight_layout()
                plt.savefig(Path(outdir, f"{strain}.png"))
                plt.close(fig)


