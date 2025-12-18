import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Generate sample data
months = [
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
]
years = ["2013"] * 10 + ["2014"] * 12 + ["2015"] * 3

# Create sample data
np.random.seed(42)
n_months = len(months)

# Cumulative budget line (blue line)
cumulative = np.linspace(5e6, 11e6, n_months) + np.random.normal(0, 2e5, n_months)

# Individual department spending (lollipops)
marketing = np.random.uniform(0.2e6, 1.2e6, n_months)
finance = np.random.uniform(0.2e6, 0.8e6, n_months)
development = np.random.uniform(0.3e6, 1.0e6, n_months)
sales = np.random.uniform(0.3e6, 1.5e6, n_months)
operations = np.random.uniform(1.5e6, 7.5e6, n_months)

# Create the plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Set background color
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Define colors matching the image
colors = {
    "Marketing": "#7cb342",  # green
    "Finance": "#d81b60",  # pink/magenta
    "Development": "#66bb6a",  # light green
    "Sales": "#ffb74d",  # orange/yellow
    "Operations": "#ff6f00",  # dark orange
}

# X-axis positions
x_pos = np.arange(n_months)

# Plot lollipops for each category
for i in x_pos:
    # Operations (tallest, orange)
    ax1.plot([i, i], [0, operations[i]], color=colors["Operations"], linewidth=2, zorder=1)
    ax1.scatter(i, operations[i], color=colors["Operations"], s=50, zorder=2)

    # Sales (yellow/orange)
    ax1.plot([i, i], [0, sales[i]], color=colors["Sales"], linewidth=2, zorder=1)
    ax1.scatter(i, sales[i], color=colors["Sales"], s=50, zorder=2)

    # Development (light green)
    ax1.plot([i, i], [0, development[i]], color=colors["Development"], linewidth=2, zorder=1)
    ax1.scatter(i, development[i], color=colors["Development"], s=50, zorder=2)

    # Finance (pink)
    ax1.plot([i, i], [0, finance[i]], color=colors["Finance"], linewidth=2, zorder=1)
    ax1.scatter(i, finance[i], color=colors["Finance"], s=50, zorder=2)

    # Marketing (green)
    ax1.plot([i, i], [0, marketing[i]], color=colors["Marketing"], linewidth=2, zorder=1)
    ax1.scatter(i, marketing[i], color=colors["Marketing"], s=50, zorder=2)

# Create second y-axis for cumulative line
ax2 = ax1.twinx()

# Plot cumulative budget line (blue line)
ax2.plot(x_pos, cumulative, color="#5b9bd5", linewidth=2.5, zorder=3)
ax2.scatter(x_pos, cumulative, color="#5b9bd5", s=40, zorder=4)

# Configure left y-axis (for lollipops)
ax1.set_ylim(0, 10e6)
ax1.set_yticks(np.arange(0, 11e6, 2e6))
ax1.set_yticklabels(["0", "2M", "4M", "6M", "8M", "10M"])
ax1.set_ylabel("", fontsize=10)
ax1.tick_params(axis="y", labelsize=9, length=0)

# Configure right y-axis (for cumulative)
ax2.set_ylim(0, 12.5e6)
ax2.set_yticks(np.arange(0, 13e6, 2.5e6))
ax2.set_yticklabels(["0", "2.5M", "5M", "7.5M", "10M", "12.5M"])
ax2.tick_params(axis="y", labelsize=9, length=0)

# Configure x-axis
ax1.set_xticks(x_pos)
ax1.set_xticklabels(months, fontsize=9)
ax1.tick_params(axis="x", length=0)
ax1.set_xlim(-0.5, n_months - 0.5)

# Add grid
ax1.grid(axis="y", color="#e0e0e0", linestyle="-", linewidth=0.8, alpha=0.7)
ax1.set_axisbelow(True)

# Remove spines
for spine in ["top", "right", "left", "bottom"]:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

# Add title and subtitle
fig.text(0.08, 0.96, "OR: Budget Spend by Type", fontsize=10, color="#666666")
fig.text(0.29, 0.96, "Mar 28, 2013 - Mar 18, 2015, by day", fontsize=9, color="#999999")
fig.text(0.08, 0.92, "$335.85m", fontsize=24, fontweight="bold", color="#000000")
fig.text(0.22, 0.925, "Current Budget Spent", fontsize=11, color="#666666")

# Create legend
legend_elements = [
    Rectangle((0, 0), 1, 1, fc=colors["Marketing"], label="Marketing"),
    Rectangle((0, 0), 1, 1, fc=colors["Finance"], label="Finance"),
    Rectangle((0, 0), 1, 1, fc=colors["Development"], label="Development"),
    Rectangle((0, 0), 1, 1, fc=colors["Sales"], label="Sales"),
    Rectangle((0, 0), 1, 1, fc=colors["Operations"], label="Operations"),
]

ax1.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False, fontsize=9)

# Adjust layout
plt.subplots_adjust(left=0.08, right=0.85, top=0.88, bottom=0.08)

plt.show()
