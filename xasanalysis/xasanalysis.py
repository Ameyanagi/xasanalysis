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
from matplotlib.figure import Figure
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
    """Read the xmu file gereated by Athena and return a larh Group object

    Args:
        file_path(str): path to the xmu file
        name(str): name of the group
        energy_col(int): column of the energy
        mu_col(int): column of the mu

    Returns:
        group(Group): larch Group object
    """

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
    """Read the data collected from QAS(transmission mode) and return a larch Group object or a list of larch Group objects

    The group will be sorted by the file name when using the glob option

    Args:
        file_path(str): path to the file
        name(str): name of the group
        energy_col(int): column of the energy
        i0_col(int): column of the i0
        it_col(int): column of the it
        use_glob(bool): use glob to read multiple files

    Returns:
        Group | list[Group]: larch Group object or a list of larch Group objects(if use_glob = True)
    """

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
    """Read the data collected from QAS(fluorescence mode) and return a larch Group object or a list of larch Group objects

    The group will be sorted by the file name when using the glob option

    Args:
        file_path(str): path to the file
        name(str): name of the group
        energy_col(int): column of the energy
        i0_col(int): column of the i0
        iff_col(int): column of the iff
        use_glob(bool): use glob to read multiple files

    Returns:
        Group | list[Group]: larch Group object or a list of larch Group objects(if use_glob=True)
    """

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
    """Read the data collected from QAS(reference reference foil) and return a larch Group object or a list of larch Group objects

    The group will be sorted by the file name when using the glob option

    Args:
        file_path(str): path to the file
        name(str): name of the group
        energy_col(int): column of the energy
        it_col(int): column of the it
        ir_col(int): column of the ir
        use_glob(bool): use glob to read multiple files

    Returns:
        Group | list[Group]: larch Group object or a list of larch Group objects(if use_glob=True)
    """
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
    """Read the data collected from QAS(Silicon Drifts Detector mode) and return a larch Group object or a list of larch Group objects

    The group will be sorted by the file name when using the glob option

    Args:
        file_path(str): path to the file
        name(str): name of the group
        roi(int): roi of the spectrum
        channels(list[int]): channels used to summed up the spectrum
        energy_col(int): column of the energy
        i0_col(int): column of the i0
        use_glob(bool): use glob to read multiple files

    Returns:
        Group | list[Group]: larch Group object or a list of larch Group objects(if use_glob=True)
    """

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

    The main strategy of calculating the metric is the following:

    1. Interpolate the spectrum to the energy grid (for example, 0.5eV spacing, this ensures that each energy contribution will be uniformly weighted)
    2. Calculate the flattened spectrum using the pre_edge function of xraylarch.
    3. Interpolate the the flattened spectrum to the energy grid
    4. Calculate the `spdist metric <https://github.com/Ameyanagi/spdist>`_ between the spectrum and the reference spectrum.
    5. Calculate the mean absolute error between the spectrum and the reference spectrum
    6. Return the average of the spdist metric and the mean absolute error

    The spdist is a average of the minimum distance between the two spectra, and the mean absolute error is the average of the absolute vertical distance between the two spectra.
    spdist metric is more sensitive towards the distance in the diagnal direnction of the two spectra. This is useful to align the spectra with similar features.

    Args:
        p(list): list of parameters. In this case, it is only the shift of the spectrum.
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        group(Group): group of the spectrum
        reference_x(np.ndarray): energy grid of the reference spectrum
        reference_y(np.ndarray): mu of the reference spectrum
        e0(float): e0 of the spectrum
        pre_edge_kws(dict): pre_edge keywords that will be passed to the pre_edge function
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

    return loss / 2


def calc_shift(
    energy_grid: np.ndarray,
    group: Group,
    reference: Group,
    pre_edge_kws: dict,
    fit_range: list[float] | None = None,
    max_shift: float = 20.0,
):
    """Calculate the shift of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric

    The main strategy of calculating the metric is the following:

    1. Interpolate the spectrum to the energy grid (for example, 0.5eV spacing, this ensures that each energy contribution will be uniformly weighted)
    2. Calculate the flattened spectrum using the pre_edge function of xraylarch.
    3. Interpolate the the flattened spectrum to the energy grid
    4. Calculate the `spdist metric <https://github.com/Ameyanagi/spdist>`_ between the spectrum and the reference spectrum.
    5. Calculate the mean absolute error between the spectrum and the reference spectrum
    6. Return the average of the spdist metric and the mean absolute error

    The spdist is a average of the minimum distance between the two spectra, and the mean absolute error is the average of the absolute vertical distance between the two spectra.
    spdist metric is more sensitive towards the distance in the diagnal direnction of the two spectra. This is useful to align the spectra with similar features.

    Args:
        energy_grid: energy grid for the metric calculation. This will be used for the interpolation of the spectrum
        group(Group): group of the spectrum
        reference(Group): group of the reference spectrum
        pre_edge_kws(dict): pre_edge keywords that will be passed to the pre_edge function
        fit_range(list): the fitting range for the metric calculation
        max_shift(float): maximum shift of the spectrum

    Returns:
        shift(float): shift of the spectrum
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
    """Class to analyze the XAS data

    This class aims to provide a simple interface to analyze the XAS data. The class is based on the larch library, which is a python library for X-ray absorption spectroscopy.
    It is a abstraction of my workflow to analyze the XAS data, which includes the following steps:

    1. Read tha data (This will be done by different functions)
    2. Read the reference data (This will done using the `xasref <https://github.com/Ameyanagi/xasref>`_ module). I am planning to replace the xasref to a better module in the future with the help of beamline scientists.
    3. Align the data to the reference data. (The energy alignment is usually done using the first derivative of the absorption coefficients. This module uses `spdist metric <https://github.com/Ameyanagi/spdist>`_ and mean absolute error as the metric to align the data)
    4. Set the e0, pre-edge, autobk, xftf parameters. (This step is not nessaary, but it is useful to use the same parameters for all the groups to get the consistent results)
    5. Normalization using the pre-edge function of xraylarch
    6. Background removal using the autobk function of xraylarch
    7. Fourier transform using the xftf function of xraylarch
    8. Plotting the results (This class provides the simple interface to plot the results)

    Attributes:
        groups(dict): dictionary of the groups
        e0(float): e0 of the spectrum
        pre_edge_kws(dict): keywords for the pre_edge function
        autobk_kws(dict): keywords for the autobk function
        xftf_kws(dict): keywords for the xftf function
        reference(Group): reference spectrum
        groups_ref(dict): dictionary of the reference groups

    Args:
        groups(dict): dictionary of the groups. Default is None.
        e0(float): e0 of the spectrum. Default is None, which allows a automatic detection.
        pre_edge_kws(dict): keywords for the pre_edge function. Default is None, which allows a automatic detection.
        autobk_kws(dict): keywords for the autobk function. Default is None, which allows a automatic detection.
        xftf_kws(dict): keywords for the xftf function. Default is None, which allows a automatic detection.
        reference(Group): reference spectrum. Default is None.
        groups_ref(dict): dictionary of the references in each group. Default is None.
    """

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
        """Add the group to the class

        This function adds the group to the groups dictionary. If the align_ref is provided, it will align the align_ref to the reference spectra that is registered.
        This function will raise an exception if the align_ref is provided but the reference is not set.

        Args:
            group(Group): group object
            name(str): name of the group
            align_ref(Group): reference group to align the group. Default is None.

        Returns:
            Self: self. This method can be chained.
        """

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
        """Add the group to the class. The list of group will be merged prior to the addition


        This function adds the group to the groups dictionary. This function will merge the list of the groups before adding to the class.

        If the align_ref is provided, it will align the align_ref to the reference spectra that is registered.
        This function will raise an exception if the align_ref is provided but the reference is not set.
        The align_ref can also be a list of groups, which will be merged before aligning the group.

        Args:
            groups(list[Group]): list of group objects
            name(str): name of the group
            align_ref(Group | list[Group]): reference group to align the group. Default is None.

        Returns:
        Self: self. This method can be chained.
        """
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
        """Register the reference spectrum to the class

        This method will register the reference spectrum to the class. The reference spectrum will be used to align the spectra to the reference spectrum.
        It can also be used for plotting the reference spectrum in the plotting methods.

        Args:
            group(Group): reference group
            ref_name(str): name of the reference spectrum

        Retruns:
            Self: self. This method can be chained.
        """

        if not hasattr(group, "label") and (ref_name is None):
            raise Exception("Please provide a group with label of ref_name")

        if ref_name:
            group.label = ref_name

        self.reference = group

        return self

    def set_reference_from_db(
        self, ref_name: str, element: str | None = None, label: str | None = None
    ) -> Self:
        """Register the reference spectrum to the class using the xasref module

        This method will registere the reference spectrum to the class using the xasref module.
        xasref is a curated list of reference spectrum that is aligned by myself, using the first derivative of the absorption coefficients.
        This module is planned to be replaced by a better module in the future, with the help of beamline scientists.
        The reference spectrum will be used to align the spectra to the reference spectrum.

        Args:
        ref_name(str): name of the reference spectrum in the xasref module
        element(str): element of the reference spectrum. This is only used for efficient loading of the dictionary and it is not nessesary.
        label(str): label of the reference spectrum. Default is None, which will use the ref_name.

        """
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
        """Calculate the shift of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric

        This method calculates the shift of the spectrum, with respect to the reference spectrum using spdist+MAE as the metric.
        The main strategy of calculating the metric is the following:

        1. Interpolate the spectrum to the energy grid (for example, 0.5eV spacing, this ensures that each energy contribution will be uniformly weighted)
        2. Calculate the flattened spectrum using the pre_edge function of xraylarch.
        3. Interpolate the the flattened spectrum to the energy grid
        4. Calculate the `spdist metric <https://github.com/Ameyanagi/spdist>`_ between the spectrum and the reference spectrum.
        5. Calculate the mean absolute error between the spectrum and the reference spectrum
        6. Return the average of the spdist metric and the mean absolute error

        The spdist is a average of the minimum distance between the two spectra, and the mean absolute error is the average of the absolute vertical distance between the two spectra.
        spdist metric is more sensitive towards the distance in the diagnal direnction of the two spectra. This is useful to align the spectra with similar features.

        Args:
            group(Group): group of the spectrum
            fit_range(list): the fitting range for the metric calculation. The default is None, which will use the e0 - 20eV to e0 + 80eV.
            max_shift(float): maximum shift of the spectrum. The default is 20.0. Please increase the value if the spectrum is totally misaligned.

        Returns:
            shift(float): shift of the spectrum. The calibrated energy is energy + shift.
        """
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
        """Remove the group from the class using the name

        Args:
            name(str): name of the group

        Returns:
            Self: self. This method can be chained.
        """
        del self.groups[name]
        return self

    def order_groups(self, order: list[str]) -> Self:
        """Reorder the groups dictionary

        Args:
            order(list): order of the groups

        Returns:
            Self: self. This method can be chained.
        """
        self.groups = {key: self.groups[key] for key in order}

    def set_e0(self, e0: float) -> Self:
        """Set the e0 of the class

        Args:
            e0(float): e0 of the class

        Returns:
            Self: self. This method can be chained.
        """
        self.e0 = e0
        return self

    def set_pre_edge_kws(self, kws) -> Self:
        """Set the pre_edge keywords

        Args:
            kws(dict): pre_edge keywords

        Returns:
            Self: self. This method can be chained.
        """
        self.pre_edge_kws = kws
        return self

    def set_autobk_kws(self, kws) -> Self:
        """Set the autobk keywords

        Args:
            kws(dict): autobk keywords

        Returns:
            Self: self. This method can be chained.
        """
        self.autobk_kws = kws
        return self

    def set_xftf_kws(self, kws) -> Self:
        """Set the xftf keywords

        Args:
            kws(dict): xftf keywords

        Returns:
            Self: self. This method can be chained.
        """
        self.xftf_kws = kws
        return self

    def pre_edge(self, calc_group: bool = True, calc_reference: bool = False) -> Self:
        """Pre-edge normalization of the groups

        This method will calculate the pre-edge of the groups.
        It is highly recommended to set the e0 and the pre_edge_kws before running this method.
        If it is not set, it will automatically detect the parameters for each group, and the parameters will not be consitent between the groups.

        Args:
            calc_group(bool): calculate the pre-edge of the groups. Default is True.
            calc_reference(bool): calculate the pre-edge of the reference. Default is False.

        Returns:
            Self: self. This method can be chained.
        """
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
        """Background removal of the groups

        This method will calculate the background removal of the groups.
        It is highly recommended to set the e0 and the autobk_kws before running this method.
        If it is not set, it will automatically detect the parameters for each group, and the parameters will not be consitent between the groups.

        Args:
            skip_pre_edge(bool): skip the pre-edge calculation. Default is True.

        Returns:
            Self: self. This method can be chained.
        """
        if not skip_pre_edge:
            self.pre_edge()

        for group in self.values():
            autobk(group, **self.autobk_kws)

        return self

    def xftf(self, skip_autobk=True) -> Self:
        """Fourier transform of the groups

        This method will calculate the Fourier transform of the groups.
        It is highly recommended to set the e0 and the xftf_kws before running this method.
        If it is not set, it will automatically detect the parameters for each group, and the parameters will not be consitent between the groups.

        Args:
            skip_autobk(bool): skip the autobk calculation. Default is True.

        Returns:
            Self: self. This method can be chained.
        """
        if not skip_autobk:
            self.autobk()

        for group in self.groups.values():
            xftf(group, **self.xftf_kws)
        return self

    def has_flat(self, groups_name: list[str] | None = None) -> bool:
        """Check if the groups have the flat attribute

            This method will check if the groups have the flat attribute.
            The check will only be done for the groups that are in the groups_name list.
            If the groups_name is None, it will check all the groups.

        Args:
            groups_name(list): list of the group names. Default is None.

        Returns:
            bool: True if all the groups have the flat attribute, False otherwise.
        """
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
        """Check if the reference groups have the flat attribute

        This method will check if the reference groups have the flat attribute.
        The check will only be done for the groups that are in the groups_name list.
        If the groups_name is None, it will check all the groups.

        Args:
            groups_name(list): list of the group names. Default is None.

        Returns:
            bool: True if all the reference groups have the flat attribute, False otherwise.
        """

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
        """Check if the groups have the chi attribute

        This mehtod will check if the groups have the chi attribute.
        The check will only be done for the groups that are in the groups_name list.
        If the groups_name is None, it will check all the groups.

        Returns:
            bool: True if all the groups have the chi attribute, False otherwise.
        """
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
        """Check if the groups have the chir attribute

        This method will check if the groups have the chir attribute.
        The check will only be done for the groups that are in the groups_name list.
        If the groups_name is None, it will check all the groups.

        Returns:
            bool: True if all the groups have the chir attribute, False otherwise.
        """
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
        """Get the e0 of the class

        This method will return the self.e0 if it is set.
        If it is not set, it will return the e0 of the first group in the groups dictionary.

        Returns:
            float: e0 of the class
        """
        if self.e0 is not None:
            return self.e0

        group = self.groups[self.keys()[0]]

        if hasattr(group, "e0") and group.e0 is not None:
            return group.e0

        return find_e0(group)

    def get_kweight(self) -> int:
        """Get the kweight of the class

        This method will return the kweight of the xftf_kws if it is set.
        If it is not set, it will return 2.
        This will be used for the plotting and the Fourier transform.

        Returns:
            int: kweight of the class
        """

        if not hasattr(self, "xftf_kws"):
            return 2

        if not hasattr(self.xftf_kws, "kweight"):
            return 2

        return self.xftf_kws["kweight"]

    def find_e0_from_derivative(self, index: int = 0) -> float:
        """Find the e0 of the group using the first derivative of the absorption coefficients

        This method will find the e0 of the group using the first derivative of the absorption coefficients.
        This method is useful to find which e0 to use for all of the spectra.

        Returns:
            float: e0 of the group
        """
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
        ax: Axes | None = None,
        fig: Figure | None = None,
    ) -> Axes:
        """Plot the flattened spectra

        This method will plot the flattend spectra of the groups.

        Args:
            groups_name(list): list of the group names. Default is None, which will plot all the groups.
            ignore_kws(list): list of the keywords to ignore in the group names. Default is None.
            plot_range(str | tuple | list): plot range of the spectra. Default is "full".
            ref(bool): plot the reference spectrum. Default is False.
            plot_legend(bool): plot the legend. Default is True.
            legend_kws(dict): legend keywords. Default is None.
            save_path(str): save path of the figure. Default is None.
            ax(Axes): axes of the plot. Default is None.

        Returns:
            Axes: axes of the plot
        """

        if groups_name is None:
            groups_name = list(self.groups.keys())

        # Creat a new figure if the ax is not provided
        if ax:
            ax_plot = ax
        else:
            fig, ax_plot = plt.subplots(figsize=(3, 3))

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
            ax_plot.plot(group.energy, group.flat, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "flat"):
                pre_edge(self.reference, **self.pre_edge_kws)

            ax_plot.plot(
                self.reference.energy,
                self.reference.flat,
                label=self.reference.label,
            )

        ax_plot.set_xlabel("Energy (eV)")
        ax_plot.set_ylabel("Normalized absorption coefficient")

        if isinstance(plot_range, list):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass
        elif plot_range.lower() == "xanes":
            e0 = self.get_e0()
            ax_plot.set_xlim(e0 - 20, e0 + 80)

        if plot_legend:
            if legend_kws:
                ax_plot.legend(**legend_kws)
            else:
                ax_plot.legend()

        if fig:
            fig.tight_layout(pad=0.5)

        # if ax is not None it will not be saved
        if ax is None and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax_plot

    def plot_flat_refs(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = True,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups_ref.keys())

        # Creat a new figure if the ax is not provided
        if ax:
            ax_plot = ax
        else:
            fig, ax_plot = plt.subplots(figsize=(3, 3))

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
            ax_plot.plot(group.energy, group.flat, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "flat"):
                pre_edge(self.reference, **self.pre_edge_kws)

            ax_plot.plot(
                self.reference.energy,
                self.reference.flat,
                label="ref: " + self.reference.label,
            )

        ax_plot.set_xlabel("Energy (eV)")
        ax_plot.set_ylabel("Normalized absorption coefficient")

        if isinstance(plot_range, list):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass
        elif plot_range.lower() == "xanes":
            e0 = self.get_e0()
            ax_plot.set_xlim(e0 - 20, e0 + 80)

        if plot_legend:
            if legend_kws:
                ax_plot.legend(**legend_kws)
            else:
                ax_plot.legend()
        if fig:
            fig.tight_layout(pad=0.5)

        # if ax is not None it will not be saved
        if ax is None and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax_plot

    def plot_k(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups.keys())

        # Creat a new figure if the ax is not provided
        if ax:
            ax_plot = ax
        else:
            fig, ax_plot = plt.subplots(figsize=(3, 3))

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
            ax_plot.plot(group.k, group.chi * group.k**kweight, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "chi"):
                pre_edge(self.reference, **self.pre_edge_kws)
                autobk(self.reference, **self.autobk_kws)

            ax_plot.plot(
                self.reference.k,
                self.reference.chi * self.reference.k**kweight,
                label=self.reference.label,
            )

        ax_plot.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
        if kweight == 0:
            ax_plot.set_ylabel(r"$\chi(k)$")
        elif kweight == 1:
            ax_plot.set_ylabel(r"$k\chi(k)$")
        elif kweight > 1:
            ax_plot.set_ylabel(r"$k^{}\chi(k)$".format(int(kweight)))

        if isinstance(plot_range, list):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass

        if plot_legend:
            if legend_kws:
                ax_plot.legend(**legend_kws)
            else:
                ax_plot.legend()

        if fig:
            fig.tight_layout(pad=0.5)

        # if ax is not None it will not be saved
        if ax is None and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax_plot

    def plot_r(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_range: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
    ) -> Axes:
        if groups_name is None:
            groups_name = list(self.groups.keys())

        # Creat a new figure if the ax is not provided
        if ax:
            ax_plot = ax
        else:
            fig, ax_plot = plt.subplots(figsize=(3, 3))

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
            ax_plot.plot(group.r, group.chir_mag, label=group_name)

        if ref and hasattr(self, "reference"):
            if not hasattr(self.reference, "chir"):
                pre_edge(self.reference, **self.pre_edge_kws)
                autobk(self.reference, **self.autobk_kws)
                xftf(self.reference, **self.xftf_kws)

            ax_plot.plot(
                self.reference.r,
                self.reference.chir_mag,
                label=self.reference.label,
            )
        ax_plot.set_xlabel(r"$R$ ($\mathrm{\AA}$)")

        ax_plot.set_ylabel(
            r"$|\chi(R)|$ ($\mathrm{\AA}^{" + str(int(-kweight - 1)) + "}$)"
        )

        if isinstance(plot_range, list):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif isinstance(plot_range, tuple):
            ax_plot.set_xlim(plot_range[0], plot_range[1])
        elif plot_range.lower() == "full":
            pass

        if plot_legend:
            if legend_kws:
                ax_plot.legend(**legend_kws)
            else:
                ax_plot.legend()

        if fig:
            fig.tight_layout(pad=0.5)

        # if ax is not None it will not be saved
        if ax is None and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)

        return ax_plot

    def plot_multi(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_erange: str | tuple[float, float] | list[float] | None = "full",
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Axes | Sequence[Axes] | None = None,
        fig: Figure | None = None,
        plot_figures: str | list[str] | None = None,
    ) -> Axes | Sequence[Axes]:

        if isinstance(plot_figures, list):
            # Convert the plot_figures to lower case for the comparison
            plot_figures = [plot_figure.lower() for plot_figure in plot_figures]

            # Define the number of rows and columns
            nrows = 1
            ncols = len(plot_figures)

            if groups_name is None:
                groups_name = list(self.groups.keys())

            # Creat a new figure if the ax is not provided
            if ax:
                ax_plot = ax
            else:
                fig, ax_plot = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3)
                )
                if ncols == 1:
                    ax_plot = np.array([ax_plot])
                ax_plot.flatten()

            # check if the group name is in the groups and not containing the keywords in the ignore_kws
            groups_name = [
                group_name
                for group_name in groups_name
                if (group_name in self.groups)
                and (
                    ignore_kws is None or not any(kw in group_name for kw in ignore_kws)
                )
            ]

            if not self.has_flat(groups_name):
                self.pre_edge()

            if not self.has_chi(groups_name):
                self.autobk()

            if not self.has_chir(groups_name):
                self.xftf()

            if ref and hasattr(self, "reference"):
                if not hasattr(self.reference, "chir"):
                    pre_edge(self.reference, **self.pre_edge_kws)
                    autobk(self.reference, **self.autobk_kws)
                    xftf(self.reference, **self.xftf_kws)

            for ax_tmp, plot_figure in zip(ax_plot, plot_figures):
                self.plot_multi(
                    groups_name=groups_name,
                    ignore_kws=ignore_kws,
                    plot_erange=plot_erange,
                    plot_krange=plot_krange,
                    plot_rrange=plot_rrange,
                    ref=ref,
                    plot_legend=plot_legend,
                    legend_kws=legend_kws,
                    save_path=None,
                    ax=ax_tmp,
                    plot_figures=plot_figure,
                )

            if isinstance(plot_legend, bool):
                if plot_legend:
                    if legend_kws:
                        ax_plot[0].legend(**legend_kws)
                    else:
                        ax_plot[0].legend()
            elif isinstance(plot_legend, list):
                plot_legend_indexes = [i for i in list if i < len(ax_plot)]

                for i in plot_legend_indexes:
                    if plot_legend[i]:
                        if legend_kws:
                            ax_plot[i].legend(**legend_kws)
                        else:
                            ax_plot[i].legend()
            elif isinstance(plot_legend, str):
                if plot_legend.lower() == "all":
                    for ax in ax_plot:
                        if legend_kws:
                            ax.legend(**legend_kws)
                        else:
                            ax.legend()

                if plot_legend.lower() in ["e", "x", "k", "r"]:
                    legend_index = plot_figures.index(plot_legend.lower())
                    if legend_kws:
                        ax_plot[legend_index].legend(**legend_kws)
                    else:
                        ax_plot[legend_index].legend()

                if plot_legend.lower() == "left":
                    legend_kws = {
                        "bbox_to_anchor": (1.05, 1),
                        "loc": "upper left",
                    }
                    ax_plot[-1].legend(**legend_kws)

            if fig:
                fig.tight_layout(pad=0.5)

            # if ax is not None it will not be saved
            if ax is None and save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=300)

            return ax_plot

        elif plot_figures is None:
            plot_figures = ["e", "k", "r"]
            return self.plot_multi(
                groups_name=groups_name,
                ignore_kws=ignore_kws,
                plot_erange=plot_erange,
                plot_krange=plot_krange,
                plot_rrange=plot_rrange,
                ref=ref,
                plot_legend=plot_legend,
                legend_kws=legend_kws,
                save_path=save_path,
                ax=ax,
                plot_figures=plot_figures,
            )

        # The plot_figures is a string

        if not isinstance(ax, Axes):
            print(ax)
            raise Exception(
                "Please provide the ax for the plot, when plot_figures is a string"
            )
        if plot_figures not in ["e", "x", "k", "r"]:
            raise Exception("Please provide a valid plot_figures, e, x, k, and r")

        ax_plot = ax

        if plot_figures == "e":
            self.plot_flat(
                groups_name=groups_name,
                ignore_kws=None,
                plot_range=plot_erange,
                ref=ref,
                plot_legend=False,
                legend_kws=None,
                save_path=None,
                ax=ax_plot,
            )
        elif plot_figures == "x":
            self.plot_flat(
                groups_name=groups_name,
                ignore_kws=None,
                plot_range="xanes",
                ref=ref,
                plot_legend=False,
                legend_kws=None,
                save_path=None,
                ax=ax_plot,
            )
        elif plot_figures == "k":
            self.plot_k(
                groups_name=groups_name,
                ignore_kws=None,
                plot_range=plot_krange,
                ref=ref,
                plot_legend=False,
                legend_kws=None,
                save_path=None,
                ax=ax_plot,
            )
        elif plot_figures == "r":
            self.plot_r(
                groups_name=groups_name,
                ignore_kws=None,
                plot_range=plot_rrange,
                ref=ref,
                plot_legend=False,
                legend_kws=legend_kws,
                save_path=None,
                ax=ax_plot,
            )

        return ax_plot

    def plot_ekr(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_erange: str | tuple[float, float] | list[float] | None = "full",
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["e", "k", "r"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=plot_erange,
            plot_krange=plot_krange,
            plot_rrange=plot_rrange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_ek(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_erange: str | tuple[float, float] | list[float] | None = "full",
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["e", "k"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=plot_erange,
            plot_krange=plot_krange,
            plot_rrange=None,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_er(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_erange: str | tuple[float, float] | list[float] | None = "full",
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["e", "r"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=plot_erange,
            plot_krange=None,
            plot_rrange=plot_rrange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_kr(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["k", "r"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_krange=plot_krange,
            plot_rrange=plot_rrange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_exkr(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_erange: str | tuple[float, float] | list[float] | None = "full",
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["e", "x", "k", "r"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=plot_erange,
            plot_krange=plot_krange,
            plot_rrange=plot_rrange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_xkr(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["x", "k", "r"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=None,
            plot_krange=plot_krange,
            plot_rrange=plot_rrange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_xk(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_krange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["x", "k"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=None,
            plot_krange=plot_krange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

    def plot_xr(
        self,
        groups_name: list[str] | None = None,
        ignore_kws: list[str] | None = None,
        plot_rrange: str | tuple[float, float] | list[float] | None = "full",
        ref: bool = False,
        plot_legend: bool | list[int] | str = True,
        legend_kws: dict | None = None,
        save_path: str | None = None,
        ax: Sequence[Axes] | None = None,
        fig: Figure | None = None,
    ) -> Sequence[Axes]:

        plot_figures = ["x", "r"]

        # Although there is a error in the static type checking, it is ensured that the return is Sequence[Axes].
        # TODO: fix to be consistent with the static type checking
        return self.plot_multi(
            groups_name=groups_name,
            ignore_kws=ignore_kws,
            plot_erange=None,
            plot_rrange=plot_rrange,
            ref=ref,
            plot_legend=plot_legend,
            legend_kws=legend_kws,
            save_path=save_path,
            ax=ax,
            fig=fig,
            plot_figures=plot_figures,
        )

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
