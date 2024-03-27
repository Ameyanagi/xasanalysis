import os
from copy import deepcopy
from glob import glob
from typing import Self, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scienceplots
import scipy.optimize as opt
import spdist
from larch import Group
from larch.io import merge_groups, read_ascii
from larch.xafs import autobk, find_e0, pre_edge, xftf, xftr
from matplotlib.axes import Axes
from xasref import get_ref_dict

plt.style.use(["science", "nature", "bright"])
font_size = 8
plt.rcParams.update({"font.size": font_size})
plt.rcParams.update({"axes.labelsize": font_size})
plt.rcParams.update({"xtick.labelsize": font_size})
plt.rcParams.update({"ytick.labelsize": font_size})
plt.rcParams.update({"legend.fontsize": font_size})


def read_xmu(
    file_path: str,
    name: str,
    energy_col: int = 0,
    mu_col: int = 1,
) -> Group:
    with open(file_path) as f:
        header = ""
        for line in f:
            if line.startswith("#"):
                header += line
            else:
                break

        data = np.loadtxt(f)

    energy = data[:, energy_col]
    mu = data[:, mu_col]
    group = Group(energy=energy, mu=mu, header=header, label=name)

    return group


def read_QAS_transmission(
    file_path: str,
    name: str,
    energy_col: int = 0,
    i0_col: int = 1,
    it_col: int = 2,
    use_glob=False,
) -> Group | list[Group]:
    if use_glob:
        files = glob(file_path)
        files.sort()

        groups = []

        for i, file in enumerate(files):
            group = read_QAS_transmission(
                file, f"{name}_{i}", energy_col, i0_col, it_col
            )
            groups.append(group)

        return groups

    with open(file_path) as f:
        header = ""
        for line in f:
            if line.startswith("#"):
                header += line
            else:
                break

        data = np.loadtxt(f)

    energy = data[:, energy_col]
    mu = -np.log(data[:, it_col] / data[:, i0_col])
    group = Group(energy=energy, mu=mu, header=header, label=name)

    return group


def read_QAS_fluorescence(
    file_path: str,
    name: str,
    energy_col: int = 0,
    i0_col: int = 1,
    iff_col: int = 4,
    use_glob=False,
) -> Group | list[Group]:
    if use_glob:
        files = glob(file_path)
        files.sort()
        groups = []
        for i, file in enumerate(files):
            group = read_QAS_fluorescence(
                file, f"{name}_{i}", energy_col, i0_col, iff_col
            )
            groups.append(group)
        return groups

    with open(file_path) as f:
        header = ""
        for line in f:
            if line.startswith("#"):
                header += line
            else:
                break

        data = np.loadtxt(f)

    energy = data[:, energy_col]
    mu = data[:, iff_col] / data[:, i0_col]
    group = Group(energy=energy, mu=mu, header=header, label=name)
    return group


def read_QAS_ref(
    file_path: str,
    name: str,
    energy_col: int = 0,
    it_col: int = 2,
    ir_col: int = 3,
    use_glob=False,
) -> Group | list[Group]:
    if use_glob:
        files = glob(file_path)
        files.sort()
        groups = []
        for i, file in enumerate(files):
            group = read_QAS_ref(file, f"{name}_{i}", energy_col, it_col, ir_col)
            groups.append(group)
        return groups

    with open(file_path) as f:
        header = ""
        for line in f:
            if line.startswith("#"):
                header += line
            else:
                break
        data = np.loadtxt(f)

    energy = data[:, energy_col]
    mu = -np.log(data[:, it_col] / data[:, ir_col])
    group = Group(energy=energy, mu=mu, header=header, label=name)
    return group


def read_QAS_SDD(
    file_path: str,
    name: str,
    roi: int = 1,
    channels: list[int] | None = None,
    energy_col: int = 0,
    i0_col: int = 1,
    use_glob=False,
):
    if use_glob:
        files = glob(file_path)
        files.sort()
        groups = []
        for i, file in enumerate(files):
            group = read_QAS_SDD(file, f"{name}_{i}", roi, channels, energy_col, i0_col)
            groups.append(group)
        return groups

    with open(file_path) as f:
        header = ""
        for line in f:
            if line.startswith("#"):
                header += line
            else:
                break
        data = np.loadtxt(f)

    if channels is None:
        channel_col_offset = [9 + (i - 1) * 4 for i in range(1, 5)]
    else:
        channel_col_offset = [9 + (i - 1) * 4 for i in channels]

    columns = [i + roi for i in channel_col_offset]

    energy = data[:, energy_col]
    mu = np.sum(data[:, columns], axis=1) / data[:, i0_col]

    group = Group(energy=energy, mu=mu, header=header, label=name)
    return group


def _residue_shift_normalize(
    p: list,
    energy_grid: np.ndarray,
    group: Group,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    e0: float,
    pre_edge_kws: dict,
    fit_range: list | None = None,
):
    """Residue to calculate the shift and scale of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric

    Args:
        p(list): shift and scale
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        group(Group): group of the spectrum
        reference(Group): group of the reference spectrum. The reference spectrum should already be noramlized to reduce the calculation time.
        e0(float): e0 of the spectrum
        pre_edge_kws(dict): pre_edge kws
        fit_range(list): the fitting range for the metric calculation

    Returns:
        residue(np.ndarray): residue of the spectrum and the reference spectrum
    """

    # print(len(spectrum_y))
    fit_group = Group(energy=deepcopy(group.energy) + p[0], mu=group.mu, e0=e0)
    pre_edge(fit_group, **pre_edge_kws)

    flat = np.interp(energy_grid, fit_group.energy, fit_group.flat)

    if fit_range:
        index = np.where((energy_grid >= fit_range[0]) & (energy_grid <= fit_range[1]))
        energy_grid = energy_grid[index]
        flat = flat[index]

    loss = spdist.spdist(energy_grid, flat, reference_x, reference_y) + np.mean(
        np.abs(flat - np.interp(energy_grid, reference_x, reference_y))
    )

    return loss


def calc_shift(
    energy_grid: np.ndarray,
    group: Group,
    reference: Group,
    pre_edge_kws: dict,
    fit_range: list[float] | None = None,
    max_shift: float = 20.0,
):
    """Calculate the shift of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric

    Args:
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        spectrum_x(np.ndarray): energy grid of the spectrum
        spectrum_y(np.ndarray): mu of the spectrum
        ref_spectrum_x(np.ndarray): energy grid of the reference spectrum
        ref_spectrum_y(np.ndarray): mu of the reference spectrum
        fit_range(list): the fitting range for the metric calculation
        max_shift(float): the maximum shift of the spectrum

    Returns:
        shift(float): shift of the spectrum
        scale(float): scale of the spectrum
        loss(float): loss of the spectrum
    """

    pre_edge(reference, **pre_edge_kws)
    e0 = reference.e0

    reference_x = reference.energy
    reference_y = reference.flat

    if fit_range:
        index = np.where(
            (reference_x >= fit_range[0] - max_shift)
            & (reference_y <= fit_range[1] + max_shift)
        )
        reference_x = reference_x[index]
        reference_y = reference_y[index]

    p0 = [0]

    print(fit_range)
    residue = lambda p: _residue_shift_normalize(
        p,
        energy_grid,
        group,
        reference_x,
        reference_y,
        e0,
        pre_edge_kws,
        fit_range,
    )

    # optimization_algorithm = opt.minimize(residue, p0, method="BFGS", )
    optimization_algorithm = opt.shgo(
        residue,
        [(-max_shift, max_shift)],
    )

    shift = optimization_algorithm.x
    loss = optimization_algorithm.fun

    return shift, loss


class XASAnalysis:
    groups: dict[str, Group]
    e0: float | None
    pre_edge_kws: dict
    autobk_kws: dict
    xftf_kws: dict
    reference: Group
    groups_ref: dict[str, Group]

    def __init__(
        self,
        groups: dict[str, Group] | None = None,
        e0: float | None = None,
        pre_edge_kws: dict | None = None,
        autobk_kws: dict | None = None,
        xftf_kws: dict | None = None,
        reference: Group | None = None,
        groups_ref: dict[str, Group] | None = None,
    ):
        if groups is None:
            self.groups = {}

        if groups_ref is None:
            self.groups_ref = {}

        self.e0 = e0

        if pre_edge_kws is None:
            self.pre_edge_kws = {}

        if autobk_kws is None:
            self.autobk_kws = {}

        if xftf_kws is None:
            self.xftf_kws = {}

        if reference:
            self.reference = reference

    def add_group(
        self, group: Group, name: str, align_ref: Group | None = None
    ) -> Self:
        group = deepcopy(group)

        if align_ref:
            shift = self.calc_shift(align_ref)
            group.energy = group.energy + shift

            self.groups_ref[name] = Group(
                energy=align_ref.energy + shift, mu=align_ref.mu, label=name + "_ref"
            )

        self.groups[name] = group

        return self

    def add_merge_group(
        self,
        groups: list[Group],
        name: str,
        align_ref: Group | list[Group] | None = None,
    ) -> Self:
        if isinstance(align_ref, Group):
            align_ref = deepcopy(align_ref)
            shift = self.calc_shift(align_ref)

            self.groups_ref[name] = Group(
                energy=align_ref.energy + shift, mu=align_ref.mu, label=name + "_ref"
            )
        elif isinstance(align_ref, list):
            align_ref = merge_groups(deepcopy(align_ref))
            shift = self.calc_shift(align_ref)

            self.groups_ref[name] = Group(
                energy=align_ref.energy + shift, mu=align_ref.mu, label=name + "_ref"
            )
        else:
            shift = 0.0

        print(len(groups))
        group = merge_groups(deepcopy(groups))
        group.energy = group.energy + shift

        self.groups[name] = group

        return self

    def set_reference(self, group: Group, ref_name: str | None) -> Self:
        if not hasattr(group, "label") and (ref_name is None):
            raise Exception("Please provide a group with label of ref_name")

        if ref_name:
            group.label = ref_name

        self.reference = group

        return self

    def set_reference_from_db(
        self, ref_name: str, element: str | None = None, label: str | None = None
    ) -> Self:
        ref_dict = get_ref_dict(element)

        group = ref_dict[ref_name]["group"]
        group.e0 = ref_dict[ref_name]["e0"]

        if label is None:
            label = ref_name

        return self.set_reference(group, label)

    def calc_shift(
        self,
        group: Group,
        fit_range: list[float] | None = None,
        max_shift: float = 20.0,
    ) -> float:
        if not hasattr(self, "reference"):
            raise Exception("Please set the reference spectrum")

        if not hasattr(self.reference, "e0"):
            raise Exception("Please set the e0 of the reference spectrum")

        if self.e0:
            e0 = self.e0
        else:
            e0 = self.reference.e0

        if fit_range is None:
            fit_range = [e0 - 20, e0 + 80]

        # The energy grid is 0.5eV spacing by default
        energy_grid = np.linspace(fit_range[0], fit_range[1], 200)

        shift, _ = calc_shift(
            energy_grid, group, self.reference, self.pre_edge_kws, fit_range, max_shift
        )
        return shift

    def remove_group(self, name: str) -> Self:
        del self.groups[name]
        return self

    def order_groups(self, order: list[str]):
        self.groups = {key: self.groups[key] for key in order}

    def set_e0(self, e0: float) -> Self:
        self.e0 = e0
        return self

    def set_pre_edge_kws(self, kws) -> Self:
        self.pre_edge_kws = kws
        return self

    def set_autobk_kws(self, kws) -> Self:
        self.autobk_kws = kws
        return self

    def set_xftf_kws(self, kws) -> Self:
        self.xftf_kws = kws
        return self

    def pre_edge(self, calc_group: bool = True, calc_reference: bool = False) -> Self:
        if calc_reference:
            for group in self.groups_ref.values():
                if self.e0 is not None:
                    group.e0 = self.e0

                pre_edge(group, **self.pre_edge_kws)

        if calc_group:
            for group in self.values():
                if self.e0 is not None:
                    group.e0 = self.e0

                pre_edge(group, **self.pre_edge_kws)

        return self

    def autobk(self, skip_pre_edge=True) -> Self:
        if not skip_pre_edge:
            self.pre_edge()

        for group in self.values():
            autobk(group, **self.autobk_kws)

        return self

    def xftf(self, skip_autobk=True) -> Self:
        if not skip_autobk:
            self.autobk()

        for group in self.groups.values():
            xftf(group, **self.xftf_kws)
        return self

    def has_flat(self, groups_name: list[str] | None = None) -> bool:
        if groups_name is None:
            groups_name = list(self.groups.keys())
        else:
            groups_name = [
                group_name for group_name in groups_name if group_name in self.groups
            ]

        return all(
            [
                hasattr(group, "flat")
                for key, group in self.groups.items()
                if key in groups_name
            ]
        )

    def has_flat_refs(self, groups_name: list[str] | None = None) -> bool:
        if groups_name is None:
            groups_name = list(self.groups_ref.keys())
        else:
            groups_name = [
                group_name
                for group_name in groups_name
                if group_name in self.groups_ref
            ]
        return all(
            [
                hasattr(group, "flat")
                for key, group in self.groups_ref.items()
                if key in groups_name
            ]
        )

    def has_chi(self, groups_name: list[str] | None = None) -> bool:
        if groups_name is None:
            groups_name = list(self.groups.keys())
        else:
            groups_name = [
                group_name for group_name in groups_name if group_name in self.groups
            ]

        return all(
            [
                hasattr(group, "chi")
                for key, group in self.groups.items()
                if key in groups_name
            ]
        )

    def has_chir(self, groups_name: list[str] | None = None) -> bool:
        if groups_name is None:
            groups_name = list(self.groups.keys())
        else:
            groups_name = [
                group_name for group_name in groups_name if group_name in self.groups
            ]
        return all(
            [
                hasattr(group, "chir")
                for key, group in self.groups.items()
                if key in groups_name
            ]
        )

    def get_e0(self) -> float:
        if self.e0 is not None:
            return self.e0

        group = self.groups[self.keys()[0]]

        if hasattr(group, "e0") and group.e0 is not None:
            return group.e0

        return find_e0(group)

    def get_kweight(self) -> int:
        if not hasattr(self, "xftf_kws"):
            return 2

        if not hasattr(self.xftf_kws, "kweight"):
            return 2

        return self.xftf_kws["kweight"]

    def find_e0_from_derivative(self, index: int = 0) -> float:
        if index > len(self.groups):
            raise Exception("Index out of range")
        return find_e0(self.groups[self.keys()[index]])

    def plot_flat(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups.keys())

        fig, ax = plt.subplots(figsize=(3, 3))

        # check if the group name is in the groups and not containing the keywords in the ignore_kws
        groups_name = [
            group_name
            for group_name in groups_name
            if (group_name in self.groups)
            and (ignore_kws is None or not any(kw in group_name for kw in ignore_kws))
        ]

        if not self.has_flat(groups_name):
            self.pre_edge()

        for i, group_name in enumerate(groups_name):
            group = self.groups[group_name]
            ax.plot(group.energy, group.flat, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "flat"):
                pre_edge(self.reference, **self.pre_edge_kws)

            ax.plot(
                self.reference.energy,
                self.reference.flat,
                label=self.reference.label,
            )

        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Normalized absorption coefficient")

        if isinstance(plot_range, list):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass
        elif plot_range.lower() == "xanes":
            e0 = self.get_e0()
            ax.set_xlim(e0 - 20, e0 + 80)

        if plot_legend:
            if legend_kws:
                ax.legend(**legend_kws)
            else:
                ax.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax

    def plot_flat_refs(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = True,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups_ref.keys())

        fig, ax = plt.subplots(figsize=(3, 3))

        # check if the group name is in the groups and not containing the keywords in the ignore_kws
        groups_name = [
            group_name
            for group_name in groups_name
            if (group_name in self.groups_ref)
            and (ignore_kws is None or not any(kw in group_name for kw in ignore_kws))
        ]

        if not self.has_flat_refs(groups_name):
            self.pre_edge(calc_reference=True, calc_group=False)

        for i, group_name in enumerate(groups_name):
            group = self.groups_ref[group_name]
            ax.plot(group.energy, group.flat, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "flat"):
                pre_edge(self.reference, **self.pre_edge_kws)

            ax.plot(
                self.reference.energy,
                self.reference.flat,
                label="ref: " + self.reference.label,
            )

        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Normalized absorption coefficient")

        if isinstance(plot_range, list):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass
        elif plot_range.lower() == "xanes":
            e0 = self.get_e0()
            ax.set_xlim(e0 - 20, e0 + 80)

        if plot_legend:
            if legend_kws:
                ax.legend(**legend_kws)
            else:
                ax.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax

    def plot_k(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups.keys())

        fig, ax = plt.subplots(figsize=(3, 3))

        # check if the group name is in the groups and not containing the keywords in the ignore_kws
        groups_name = [
            group_name
            for group_name in groups_name
            if (group_name in self.groups)
            and (ignore_kws is None or not any(kw in group_name for kw in ignore_kws))
        ]

        if not self.has_chi(groups_name):
            self.autobk()

        kweight = self.get_kweight()

        for i, group_name in enumerate(groups_name):
            group = self.groups[group_name]
            ax.plot(group.k, group.chi * group.k**kweight, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "chi"):
                pre_edge(self.reference, **self.pre_edge_kws)
                autobk(self.reference, **self.autobk_kws)

            ax.plot(
                self.reference.k,
                self.reference.chi * self.reference.k**kweight,
                label=self.reference.label,
            )

        ax.set_xlabel("$k$ ($\mathrm{\AA}^-1$)")
        if kweight == 0:
            ax.set_ylabel("$\chi(k)$")
        elif kweight == 1:
            ax.set_ylabel("$k\chi(k)$")
        elif kweight > 1:
            ax.set_ylabel("$k^{}\chi(k)$".format(int(kweight)))

        if isinstance(plot_range, list):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass

        if plot_legend:
            if legend_kws:
                ax.legend(**legend_kws)
            else:
                ax.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax

    def plot_r(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups.keys())

        fig, ax = plt.subplots(figsize=(3, 3))

        # check if the group name is in the groups and not containing the keywords in the ignore_kws
        groups_name = [
            group_name
            for group_name in groups_name
            if (group_name in self.groups)
            and (ignore_kws is None or not any(kw in group_name for kw in ignore_kws))
        ]

        if not self.has_chir(groups_name):
            self.xftf()

        kweight = self.get_kweight()

        for i, group_name in enumerate(groups_name):
            group = self.groups[group_name]
            ax.plot(group.r, group.chir_mag, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "chir"):
                pre_edge(self.reference, **self.pre_edge_kws)
                autobk(self.reference, **self.autobk_kws)
                xftf(self.reference, **self.xftf_kws)

            ax.plot(
                self.reference.r,
                self.reference.chir_mag,
                label=self.reference.label,
            )
        ax.set_xlabel("$R$ ($\mathrm{\AA}$)")
        # TODO: Add the correct ylabel
        ax.set_ylabel("$|\chi(R)|$ ($\mathrm{\AA}^{-3}$)")

        # ax.set_xlabel("$k$ ($\mathrm{\AA}^-1$)")
        # if kweight == 0:
        #     ax.set_ylabel("$\chi(k)$")
        # elif kweight == 1:
        #     ax.set_ylabel("$k\chi(k)$")
        # elif kweight > 1:
        #     ax.set_ylabel("$k^{}\chi(k)$".format(int(kweight)))

        if isinstance(plot_range, list):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass

        if plot_legend:
            if legend_kws:
                ax.legend(**legend_kws)
            else:
                ax.legend()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax

    def __dir__(self):
        return list(self.groups.keys())

    def keys(self):
        return list(self.groups.keys())

    def values(self):
        return self.groups.values()

    def items(self):
        return self.groups.items()

    def __getitem__(self, key: str):
        return self.groups[key]

    def __setitem__(self, key: str, value: Group):
        self.groups[key] = value

    def __delitem__(self, key: str):
        del self.groups[key]

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)
