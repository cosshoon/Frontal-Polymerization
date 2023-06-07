import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rc("font", family="Times New Roman")

# read in data

df = pd.read_excel(
    "/Users/gavin/FP_Printing_Modeling/printing/composites/roller_printer/composite_print_lr_nondim.xlsx",
    header=2,
).dropna()


# helpers to find the roller radius from a given contact length
def lr(R):

    hgap = 3
    phi1 = 0.3
    phi2 = 0.4
    alpha = np.arccos(1 - (hgap / (2 * R)) * (phi2 / phi1 - 1))
    lr = alpha * R

    return lr


def lr_inv(r1, r2, desired_lr):

    radius_range = np.linspace(r1, r2, int(1e7))
    idx = np.argmin(np.abs(lr(radius_range) - desired_lr))
    return radius_range[idx]


print(lr_inv(15.9, 24.9, 4.5))


# plot contours for alpha as a function of contact length and temp for each Vin

fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 60))
plt.subplots_adjust(hspace=0.3)
for i, vin in enumerate(np.unique(df["V_in [mm/s]"].values)):

    if vin == 5.0:
        continue

    curr = df[df["V_in [mm/s]"] == vin]
    lr = curr["l_R [mm]"].values
    temp = curr["T_R [degC]"].values
    alpha = curr["alpha_max_midline"].values

    csf = axes[i].tricontourf(
        temp,
        lr,
        alpha,
        levels=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
    )

    cs = axes[i].tricontour(
        temp,
        lr,
        alpha,
        levels=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
        linewidths=1,
        colors="black",
    )

    axes[i].clabel(cs, cs.levels, colors="black")
    axes[i].set_title(
        r"$\alpha_{max}$ for $V_{in}$ = " + str(vin) + " mm/s and $H_{gap}$ = 3 mm",
        fontsize=16,
    )

    cb = fig.colorbar(csf, pad=0.1)
    cb.set_label(r"$\alpha_{max}$", loc="center", rotation=0, labelpad=20)

    axes[i].set_xlabel("Temperature ($\degree$C)", fontsize=13)
    axes[i].set_ylabel("Contact Length (mm)", fontsize=13, labelpad=10)

    xticks = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    yticks = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels(xticks, fontsize=12)
    axes[i].set_yticks(yticks)
    axes[i].set_yticklabels(yticks, fontsize=12)
    axes[i].grid(True, color="black", lw=0.5, alpha=0.30)

    ax2 = axes[i].twinx()
    ax2.set_ylim(2, 5)
    yticks2 = [
        3.9,
        6.2,
        8.9,
        12.2,
        15.9,
        20.2,
        24.9,
    ]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks2, fontsize=12)
    ax2.set_ylabel("Roller Radius (mm)", rotation=270, labelpad=14, fontsize=12)

plt.savefig("alpha_temp_lr_contour.png", dpi=300, bbox_inches="tight")


# plot contours for Q as function of contact length and temp for each Vin


fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 60))
plt.subplots_adjust(hspace=0.3)

for i, vin in enumerate(np.unique(df["V_in [mm/s]"].values)):

    if vin == 5.0:
        continue

    curr = df[df["V_in [mm/s]"] == vin]
    lr = curr["l_R [mm]"].values
    temp = curr["T_R [degC]"].values
    Q = -curr["Q [W/m]"].values / 1000

    csf = axes[i].tricontourf(
        temp,
        lr,
        Q,
    )

    cs = axes[i].tricontour(
        temp,
        lr,
        Q,
        linewidths=1,
        colors="black",
    )

    axes[i].clabel(cs, cs.levels, colors="black")
    axes[i].set_title(
        r"Power Input for $V_{in}$ = " + str(vin) + " mm/s and $H_{gap}$ = 3 mm",
        fontsize=16,
    )

    cb = fig.colorbar(csf, pad=0.1)
    cb.set_label("$Q$" + "\n" + "$[kW/m]$", loc="center", rotation=0, labelpad=20)

    axes[i].set_xlabel("Temperature ($\degree$C)", fontsize=13)
    axes[i].set_ylabel("Contact Length (mm)", fontsize=13, labelpad=10)

    xticks = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    yticks = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels(xticks, fontsize=12)
    axes[i].set_yticks(yticks)
    axes[i].set_yticklabels(yticks, fontsize=12)
    axes[i].grid(True, color="black", lw=0.5, alpha=0.30)

    ax2 = axes[i].twinx()
    ax2.set_ylim(2, 5)
    yticks2 = [
        3.9,
        6.2,
        8.9,
        12.2,
        15.9,
        20.2,
        24.9,
    ]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticks2, fontsize=12)
    ax2.set_ylabel("Roller Radius (mm)", rotation=270, labelpad=14, fontsize=12)

plt.savefig("Q_temp_lr_contour.png", dpi=300, bbox_inches="tight")


# plot front locations vs temperature for each Vin

fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(10, 60))
plt.subplots_adjust(hspace=0.3)

for i, vin in enumerate(np.unique(df["V_in [mm/s]"].values)):

    for lr in [2, 3, 4, 5]:

        lr_data = df[(np.isclose(df["l_R [mm]"], lr)) & (df["V_in [mm/s]"] == vin)]
        lf = lr_data["L_front [mm]"].values
        temp = lr_data["T_R [degC]"].values

        axes[i].plot(temp, lf, marker="o", label=r"$\ell_{r}$ = " + str(lr) + " mm")
        axes[i].set_xlabel("Temperature ($\degree$C)", fontsize=13)
        axes[i].set_ylabel("Front Location (mm)", fontsize=13, labelpad=10)
        xticks = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        yticks = [-6, -4, -2, 0, 2, 4, 6]
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xticks, fontsize=12)
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels(yticks, fontsize=12)
        axes[i].grid(True, color="black", lw=0.5, alpha=0.30)

        axes[i].set_title(
            r"Front Location for $V_{in}$ = " + str(vin) + " mm/s and $H_{gap}$ = 3 mm",
            fontsize=16,
        )

    axes[i].legend(fontsize=12)

plt.savefig("Lf_v_temp.png", dpi=300, bbox_inches="tight")
