import os
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib as mpl
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from shapely.geometry import MultiPoint, Point
mpl.rcParams['font.family'] = 'Helvetica'


def make_summary_tables(df_gpd):
    party_share = ['Con PC', 'Lab PC', 'Lib PC', 'Brx PC']
    crosstab_var = ['ASMR_f',
                    'ASMR_m',
                    'Income',
                    'Employment',
                    'Education, skills and training',
                    'Health deprivation and disability',
                    'IMD rank 2019'
                   ]

    table_rho = pd.DataFrame(columns=party_share, index=crosstab_var)
    table_p = pd.DataFrame(columns=party_share, index=crosstab_var)
    for col in party_share:
        for index in crosstab_var:
            if '%' in index:
                df_gpd[index] = df_gpd[index].rank(ascending=True)
            rho, p = spearmanr(df_gpd[col].rank(ascending=True),
                               df_gpd[index]
                               )
            table_rho.loc[index, col] = rho
            table_p.loc[index, col] = p
    return table_rho, table_p


def plot_over_time(df_sii, df_hid):
    colors = ['#E4003B', '#0087DC', '#4a6741']
    fig = plt.figure(figsize=(10, 4.75))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0:, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax2_twinx = ax2.twinx()
    ax3 = plt.subplot(gs[1, 1])
    ax3_twinx = ax3.twinx()

    females = df_sii[df_sii['sex'] == 'Females']
    ax1.errorbar(
        females.index,
        females['value'],
        yerr=[females['value'] - females['lower95ci'],
              females['upper95ci'] - females['value']],
        fmt='o',
        label='Females',
        markersize=7,
        color=colors[0],
        markeredgecolor='k',
        ecolor='k',
        linewidth=.5,
        capsize=6
    )

    males = df_sii[df_sii['sex'] == 'Males']
    ax1.errorbar(
        males.index,
        males['value'],
        yerr=[males['value'] - males['lower95ci'],
              males['upper95ci'] - males['value']],
        fmt='o',
        label='Males',
        markersize=7,
        color=colors[1],
        markeredgecolor='k',
        ecolor='k',
        linewidth=.5,
        capsize=6
    )

    legend_elements1 = [
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors[1],
               markeredgecolor='k',
               markersize=6,
               label='Male SII'),
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors[0],
               markeredgecolor='k',
               markersize=6, label='Female SII'),
        Line2D(
            [0, 0.25], [0, 0],
            linewidth=1,
            marker=None,
            color='k',
            label='95% CI'
        )
    ]
    ax1.legend(handles=legend_elements1,
               loc='lower right',
               frameon=True,
               fontsize=9,
               framealpha=1,
               facecolor='w',
               edgecolor=(0, 0, 0, 1),
               ncols=1
               )

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7))

    df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Male') &
           (df_hid['Category'] == '10 = least deprived')][['Value']].plot(ax=ax2_twinx,
                                                                          linewidth=0.5,
                                                                          markerfacecolor=colors[0],
                                                                          color='k',
                                                                          marker='o',
                                                                          markeredgecolor='k',
                                                                          legend=False,
                                                                          )

    df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Male') &
           (df_hid['Category'] == '5')][['Value']].plot(ax=ax2_twinx,
                                                        linewidth=0.5,
                                                        markerfacecolor=colors[1],
                                                        color='k',
                                                        marker='o',
                                                        markeredgecolor='k',
                                                        legend=False,
                                                        )

    df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Male') &
           (df_hid['Category'] == '01 = most deprived')][['Value']].plot(ax=ax2_twinx,
                                                                         linewidth=0.5,
                                                                         markerfacecolor=colors[2],
                                                                         color='k',
                                                                         marker='o',
                                                                         markeredgecolor='k',
                                                                         legend=False,
                                                                         )

    df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Female') &
           (df_hid['Category'] == '10 = least deprived')][['Value']].plot(ax=ax3_twinx,
                                                                          linewidth=0.5,
                                                                          markerfacecolor=colors[0],
                                                                          color='k',
                                                                          marker='o',
                                                                          markeredgecolor='k',
                                                                          legend=False,
                                                                          )

    df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Female') &
           (df_hid['Category'] == '5')][['Value']].plot(ax=ax3_twinx,
                                                        linewidth=0.5,
                                                        markerfacecolor=colors[1],
                                                        color='k',
                                                        marker='o',
                                                        markeredgecolor='k',
                                                        legend=False,
                                                        )

    df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Female') &
           (df_hid['Category'] == '01 = most deprived')][['Value']].plot(ax=ax3_twinx,
                                                                         linewidth=0.5,
                                                                         markerfacecolor=colors[2],
                                                                         color='k',
                                                                         marker='o',
                                                                         markeredgecolor='k',
                                                                         legend=False,
                                                                         )

    ax2_twinx.grid(which="major",
                   linestyle='--',
                   axis='y',
                   alpha=0.3
                   )
    ax2.grid(which="major",
             linestyle='--',
             axis='x',
             alpha=0.3
             )

    ax3_twinx.grid(which="major",
                   linestyle='--',
                   axis='y',
                   alpha=0.3
                   )
    ax3.grid(which="major",
             linestyle='--',
             axis='x',
             alpha=0.3
             )
    ax1.grid(which='major',
             linestyle='--',
             axis='both',
             alpha=0.3
             )

    ax1.set_title('a.', fontsize=16, loc='left')
    ax2_twinx.set_title('b.', fontsize=16, loc='left')
    ax3_twinx.set_title('c.', fontsize=16, loc='left')

    ax1.set_ylabel('Slope Index of Inequality\n(Life Expectancy at Birth Differential)')
    ax2_twinx.set_ylabel('Male Healthy LE\n(at Birth)')
    ax3_twinx.set_ylabel('Female Healthy LE\n(at Birth)')

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x',
                       rotation=0,
                       labelsize=9)

    ax2_twinx.set_ylim(43.5, 72.5)
    ax3_twinx.set_ylim(43.5, 72.5)

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])

    ax3.set_yticklabels([])
    ax3.set_yticks([])

    sns.despine(ax=ax1)
    sns.despine(ax=ax2, left=True, right=False)
    sns.despine(ax=ax2_twinx, left=True)
    sns.despine(ax=ax3, left=True, right=False)
    sns.despine(ax=ax3_twinx, left=True)

    legend_elements3 = [
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors[0],
               markeredgecolor='k',
               markersize=6,
               label='Least Deprived'),
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors[1],
               markeredgecolor='k',
               markersize=6, label='5th Decile'),
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors[2],
               markeredgecolor='k',
               markersize=6, label='Most Deprived'),
    ]

    ax3_twinx.legend(handles=legend_elements3,
                     loc='lower center',
                     frameon=True,
                     fontsize=9,
                     framealpha=1,
                     facecolor=(1, 1, 1, 1),
                     edgecolor=(0, 0, 0, 1),
                     ncols=3
                     )

    plt.savefig(os.path.join(os.getcwd(),
                             '..',
                             'output',
                             'health_over_time_and_deprivation.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(os.getcwd(),
                             '..',
                             'output',
                             'health_over_time_and_deprivation.svg'),
                bbox_inches='tight')

    plt.tight_layout()


def make_sii():
    df = pd.read_excel(os.path.join(os.getcwd(),
                                    '..',
                                    'data',
                                    'raw',
                                    'life_expectancy',
                                    'trends_in_sii.xlsx')
                      )
    df['timeperiod'] = df['timeperiod'].str.replace(' - ', '-')
    return df.set_index('timeperiod')


def make_hid():
    df_hid = pd.read_csv(os.path.join(os.getcwd(),
                                      '..',
                                      'data',
                                      'raw',
                                      'life_expectancy',
                                      'HID_data_Life expectancy and mortality.csv')
                         )
    df_hid['Time period'] = df_hid['Time period'].str.replace(' - ', '-')
    return df_hid.set_index('Time period')


def minimum_bounding_circle(polygon):
    points = list(polygon.exterior.coords)
    multi_point = MultiPoint(points)
    center = multi_point.centroid
    radius = 0

    for point in multi_point.geoms:
        distance = center.distance(point)
        if distance > radius:
            radius = distance

    return center, radius


class BivariateChoroplethPlotter:
    """Bivariate choropleth plotter"""

    def __init__(self, percentile_limits):
        self.percentile_limits = percentile_limits
        self.num_groups = len(percentile_limits)
        self.alpha_value = 0.85

        self.light_gray, self.green, self.blue, self.dark_blue = self.define_corner_colors()
        self.color_list_hex = self.create_color_list()

    def hex_to_rgb_color(self, hex_code):
        rgb_values = [int(hex_code[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
        return rgb_values

    def define_corner_colors(self):
        light_gray = self.hex_to_rgb_color('#e8e8e8')
        green = self.hex_to_rgb_color('#6c83b5')
        blue = self.hex_to_rgb_color('#73ae80')
        dark_blue = self.hex_to_rgb_color('#2a5a5b')
        return light_gray, green, blue, dark_blue

    def create_color_list(self):
        light_gray_to_green = []
        blue_to_dark_blue = []
        color_list = []

        for i in range(self.num_groups):
            light_gray_to_green.append([self.light_gray[j] + (self.green[j] - self.light_gray[j]) * i / (self.num_groups - 1) for j in range(3)])
            blue_to_dark_blue.append([self.blue[j] + (self.dark_blue[j] - self.blue[j]) * i / (self.num_groups - 1) for j in range(3)])

        for i in range(self.num_groups):
            for j in range(self.num_groups):
                color_list.append([light_gray_to_green[i][k] + (blue_to_dark_blue[i][k] - light_gray_to_green[i][k]) * j / (self.num_groups - 1) for k in range(3)])

        color_list_hex = ['#%02x%02x%02x' % tuple(int(c * 255) for c in color) for color in color_list]
        return color_list_hex

    def get_bivariate_color(self, p1, p2):
        if p1 >= 0 and p2 >= 0:
            i = next(i for i, pb in enumerate(self.percentile_limits) if p1 <= pb)
            j = next(j for j, pb in enumerate(self.percentile_limits) if p2 <= pb)
            return self.color_list_hex[i * self.num_groups + j]
        else:
            return '#cccccc'


    def plot_bivariate_choropleth(self, geometry, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig = ax.get_figure()

        geometry['color_bivariate'] = [self.get_bivariate_color(p1, p2) for p1,
                                       p2 in zip(geometry['column1'].values,
                                                 geometry['column2'].values)]
        geometry.plot(ax=ax,
                      color=geometry['color_bivariate'],
                      alpha=self.alpha_value,
                      legend=False,
                      edgecolor='w',
                      linewidth=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot_inset_legend(self, ax, yaxis_label):
        ax_inset = ax.inset_axes([0.0125, 0.275, 0.3, 0.3])
        ax_inset.set_aspect('equal', adjustable='box')
        x_ticks = [0]
        y_ticks = [0]

        for i, percentile_bound_p1 in enumerate(self.percentile_limits):
            for j, percentile_bound_p2 in enumerate(self.percentile_limits):
                rect = plt.Rectangle((i, j), 1, 1,
                                     edgecolor = 'k',
                                     linewidth=0.5,
                                     facecolor=self.color_list_hex[i * self.num_groups + j],
                                     alpha=self.alpha_value)
                ax_inset.add_patch(rect)
                if i == 0:
                    y_ticks.append(percentile_bound_p2)
            x_ticks.append(percentile_bound_p1)

        ax_inset.set_xlim([0, len(self.percentile_limits)])
        ax_inset.set_ylim([0, len(self.percentile_limits)])
        ax_inset.set_xticks(list(range(len(self.percentile_limits) + 1)),
                            x_ticks, fontsize=15)
        ax_inset.set_xlabel('Labour Vote Percentile\n(2019)', fontsize=16)
        ax_inset.set_yticks(list(range(len(self.percentile_limits) + 1)),
                            y_ticks, fontsize=15)
        ax_inset.set_ylabel(yaxis_label, fontsize=16)

def plot_bivariate_choropleth_map(geometry):
    percentile_limits = [0.2, 0.4, 0.6, 0.8, 1.0]
    plotter = BivariateChoroplethPlotter(percentile_limits)
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))  # Creating two subplots side by side

    lab_prevHD_rho, lab_prevHD_p  = spearmanr(geometry['Lab PC'].rank(ascending=True),
                                              geometry['Health deprivation and disability'].rank(ascending=True))
    lab_prevHD_rho = np.round(lab_prevHD_rho, 3)
    print(f'Labour vote share vs Health Deprivation rank correlation coefficient: {lab_prevHD_rho}, p-value {lab_prevHD_p}')

    lab_prevIMD_rho, lab_prevIMD_p  = spearmanr(geometry['Lab PC'].rank(ascending=True),
                                                geometry['IMD rank 2019'].rank(ascending=True))
    lab_prevIMD_rho = np.round(lab_prevIMD_rho, 3)
    print(f'Labour vote share vs IMD rank correlation coefficient: {lab_prevIMD_rho}, p-value {lab_prevIMD_p}')

    geometry['column1'] = geometry['Lab PC'].rank(ascending=True)/533
    geometry['column2'] = geometry['Health deprivation and disability'].rank(ascending=True)/533
    geometry['column3'] = geometry['IMD rank 2019'].rank(ascending=True)/533

    plotter.plot_bivariate_choropleth(geometry, ax=axs[0])
    plotter.plot_inset_legend(axs[0], 'Health Deprivation and Disability\n(Rank Percentile)')
    axs[0].set_xlim(125000, 660000)
    axs[0].set_ylim(10000, 675000)
    axs[0].set_title('a.', fontsize=35, loc='left', y=0.965, x=.0)
    sns.despine(ax=axs[0], left=True, right=True, top=True, bottom=True)
    axs[0].annotate('N', xy=(0.8, 0.95), xytext=(0.8, 0.95-.125),
                    arrowprops=dict(facecolor='black',
                                    width=2.5,
                                    headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=axs[0].transAxes)

    geometry['column2'] = geometry['column3']
    plotter.plot_bivariate_choropleth(geometry, ax=axs[1])
    plotter.plot_inset_legend(axs[1], 'Index of Multiple Deprivation\n(Rank Percentile)')
    axs[1].set_xlim(125000, 660000)
    axs[1].set_ylim(10000, 675000)
    axs[1].set_title('b.', fontsize=35, loc='left', y=0.965, x=.0)
    sns.despine(ax=axs[1], left=True, right=True, top=True, bottom=True)

    axs[1].annotate('N', xy=(0.8, 0.95), xytext=(0.8, 0.95-.125),
                    arrowprops=dict(facecolor='black', width=2.5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=axs[1].transAxes)
    starmer = geometry[geometry['ConstituencyName'].str.contains('Pancras',
                                                                 regex=False)]
    sunak = geometry[geometry['ConstituencyName'].str.contains('Richmond (Yorks)',
                                                               regex=False)]
    speaker = geometry[geometry['ConstituencyName'].str.contains('Chorley',
                                                                 regex=False)]
    for ax in [axs[0], axs[1]]:
        convex_hull = sunak['geometry'].convex_hull
        center, radius = minimum_bounding_circle(convex_hull.iloc[0])
        circle = Point(center).buffer(radius*1)
        gpd.GeoSeries([circle]).plot(color=(1, 1, 1, 0.2),
                                     ax=ax,
                                     edgecolor=(0,0,0,1),
                                     linewidth=1.5
                                    )

        ax.annotate('Richmond (Yorks)\nRishi Sunak',
                    xy=(center.x, center.y),
                    xytext=(center.x - 200000, center.y),
                    ha='center',
                    va='center',
                    fontsize=16,
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle="arc3,rad=-.45",
                                    color='black',
                                    mutation_scale=30,
                                    lw=1.5)
                   )

        convex_hull = starmer['geometry'].convex_hull
        center, radius = minimum_bounding_circle(convex_hull.iloc[0])
        circle = Point(center).buffer(radius*1.3)
        gpd.GeoSeries([circle]).plot(color=(1, 1, 1, 0.2),
                                     ax=ax,
                                     edgecolor=(0,0,0,1),
                                     linewidth=1.5
                                    )

        ax.annotate('Holborn And St Pancras\nKeir Starmer',
                    xy=(center.x, center.y),
                    xytext=(center.x, center.y-140000),
                    ha='center',
                    va='center',
                    fontsize=16,
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle="arc3,rad=-.45",
                                    color='black',
                                    mutation_scale=30,
                                    lw=1.5)
                   )


        convex_hull = speaker['geometry'].convex_hull
        center, radius = minimum_bounding_circle(convex_hull.iloc[0])
        circle = Point(center).buffer(radius*1.2)
        gpd.GeoSeries([circle]).plot(color=(1, 1, 1, 0.2),
                                     ax=ax,
                                     edgecolor=(0,0,0,1),
                                     linewidth=1.5
                                    )

        ax.annotate('Chorley\nSpeaker of the House',
                    xy=(center.x, center.y),
                    xytext=(center.x - 135000, center.y),
                    ha='center',
                    va='center',
                    fontsize=16,
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle="arc3,rad=-.45",
                                    color='black',
                                    mutation_scale=30,
                                    lw=1.5)
                   )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(os.path.join(os.getcwd(),
                             '..',
                             'output',
                             'deprivation_by_constituency_bivariate_choropleth.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(os.getcwd(),
                             '..',
                             'output',
                             'deprivation_by_constituency_bivariate_choropleth.svg'),
                bbox_inches='tight')


def make_df_gpd(df_m):
    border = gpd.read_file(os.path.join(os.getcwd(),
                                        '..',
                                        'data',
                                        'shapefile',
                                        'WPC_Dec_2019_GCB_UK.shp')
                                        )
    df_gpd = pd.merge(df_m,
                      border,
                      how='left',
                      left_on='ONSConstID',
                      right_on='pcon19cd'
                      )
    df_gpd = gpd.GeoDataFrame(df_gpd)
    df_gpd = df_gpd.set_geometry("geometry")
    return df_gpd


def plot_scatters(df_m):
    gs = (grid_spec.GridSpec(1,2))
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    df_m = df_m.copy()

    color_mapping = {
        'Lab': '#E4003B',
        'Con': '#0087DC',
        'Other': 'lightgrey'
    }

    df_m.loc[:, 'color'] = df_m['first_party_3'].map(color_mapping)

    ax1.scatter(df_m['Lab PC'],
                df_m['ASMR_f'],
                color=df_m['color'],
                edgecolor='k'
               )
    ax2.scatter(df_m['Lab PC'],
                df_m['ASMR_m'],
                color=df_m['color'],
                edgecolor='k'
               )


    ax1.set_xlabel('Labour Vote Share', fontsize=14)
    ax2.set_xlabel('Labour Vote Share', fontsize=14)
    ax1.set_ylabel('ASMR (Female)', fontsize=14)
    ax2.set_ylabel('ASMR (Male)', fontsize=14)

    legend_elements2 = [
        Patch(facecolor=color_mapping['Con'], edgecolor=(0,0,0,1),
              label=r'Conservative'),
        Patch(facecolor=color_mapping['Lab'], edgecolor=(0,0,0,1),
              label=r'Labour'),
        Patch(facecolor=color_mapping['Other'], edgecolor=(0,0,0,1),
              label=r'Other'),
    ]
    legend = ax1.legend(handles=legend_elements2,
                        loc='upper left',
                        frameon=True,
                        fontsize=10,
                        framealpha=1,
                        facecolor='w',
                        edgecolor=(0, 0, 0, 1),
                        ncols=1
                       )

    for ax, title in zip([ax1, ax2], ['a.', 'b.',]):
        ax.grid(which="both", linestyle='--', alpha=0.3)
        ax.set_title(title, loc='left', fontsize=20, y=1.0)

    lab_asmr_f_r, lab_asmr_f_p  = pearsonr(df_m['Lab PC'], df_m['ASMR_f'])
    lab_asmr_f_r = np.round(lab_asmr_f_r, 4)

    at = AnchoredText(
        r"$r$ = " + str(lab_asmr_f_r), prop=dict(size=13),
        frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)
    print(f'Labour vote share vs ASMR (F) pearsons r: {lab_asmr_f_r}, p-value {lab_asmr_f_p}')

    lab_asmr_m_r, lab_asmr_m_p = pearsonr(df_m['Lab PC'], df_m['ASMR_m'])
    lab_asmr_m_r = np.round(lab_asmr_m_r, 3)
    at = AnchoredText(
        r"$r$ = " + str(lab_asmr_m_r), prop=dict(size=13),
        frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)
    print(f'Labour vote share vs ASMR (M) pearsons r: {lab_asmr_m_r}, p-value {lab_asmr_m_p}')

    starmer = df_m[df_m['ConstituencyName'].str.contains('Pancras',
                                                       regex=False)]
    sunak = df_m[df_m['ConstituencyName'].str.contains('Richmond (Yorks)',
                                                       regex=False)]
    sunak = sunak[['Lab PC', 'ASMR_f', 'ASMR_m', 'Health deprivation and disability', 'IMD rank 2019']]
    starmer = starmer[['Lab PC', 'ASMR_f', 'ASMR_m', 'Health deprivation and disability', 'IMD rank 2019']]

    ax1.annotate('Richmond\n(Yorks)',
                 xy=(sunak['Lab PC'].iloc[0], sunak['ASMR_f'].iloc[0]),
                 xytext=(sunak['Lab PC'].iloc[0], sunak['ASMR_f'].iloc[0] + 230),
                 ha='center',
                 va='bottom',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=.45",
                                 color='black',
                                 mutation_scale=30,
                                 lw=1.5)
                )
    ax1.annotate('Holborn and\nSt Pancras',
                 xy=(starmer['Lab PC'].iloc[0], starmer['ASMR_f'].iloc[0]),
                 xytext=(starmer['Lab PC'].iloc[0], starmer['ASMR_f'].iloc[0] + 255),
                 ha='center',
                 va='bottom',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=-.45",
                                 color='black',
                                 mutation_scale=30,
                                 lw=1.5)
                )

    ax2.annotate('Richmond\n(Yorks)',
                 xy=(sunak['Lab PC'].iloc[0], sunak['ASMR_m'].iloc[0]),
                 xytext=(sunak['Lab PC'].iloc[0], sunak['ASMR_m'].iloc[0] + 340),
                 ha='center',
                 va='bottom',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=.45",
                                 color='black',
                                 mutation_scale=30,
                                 lw=1.5)
                )
    ax2.annotate('Holborn and\nSt Pancras',
                 xy=(starmer['Lab PC'].iloc[0], starmer['ASMR_m'].iloc[0]),
                 xytext=(starmer['Lab PC'].iloc[0], starmer['ASMR_m'].iloc[0] + 370),
                 ha='center',
                 va='bottom',
                 arrowprops=dict(arrowstyle='->',
                                 connectionstyle="arc3,rad=-.45",
                                 color='black',
                                 mutation_scale=30,
                                 lw=1.5)
                )
    sns.despine()
    plt.savefig(os.path.join(os.getcwd(), '..', 'output', 'health_by_constituency_scatter.pdf'),
                bbox_inches = 'tight')
    plt.savefig(os.path.join(os.getcwd(), '..', 'output', 'health_by_constituency_scatter.svg'),
                bbox_inches = 'tight')


def make_merged():
    mortality_fname = 'prematuredeathsandagestandardisedmortalityratesbyparlimentaryconsituencyandsexin2021.xlsx'
    df_mortality_m = pd.read_excel(os.path.join('..', 'data', 'raw', 'mortality',
                                                mortality_fname),
                                   sheet_name='1', skiprows=4
                                  )

    df_mortality_f = pd.read_excel(os.path.join('..', 'data', 'raw', 'mortality',
                                                mortality_fname),
                                   sheet_name='2', skiprows=4
                                  )
    df_deprivation = pd.read_excel(os.path.join('..', 'data', 'raw', 'deprivation',
                                                'deprivation-dashboard.xlsx'),
                                   sheet_name='Data constituencies'
                                  )
    df_voting = pd.read_excel(os.path.join('..', 'data', 'raw', 'voting',
                                           'HoC-GE2019-results-by-constituency.xlsx'),
                              sheet_name='Data'
                             )
    df_mortality_m = df_mortality_m.rename({'ASMR': 'ASMR_m'},
                                           axis=1)
    df_mortality_f = df_mortality_f.rename({'ASMR': 'ASMR_f'},
                                           axis=1)
    asmr = pd.merge(df_mortality_m, df_mortality_f,
                    how='left',
                    right_on='Parliamentary Constituency ',
                    left_on='Parliamentary Constituency ')
    asmr = asmr[['Parliamentary Constituency ',
                 'ASMR_f',
                 'ASMR_m']]

    df_voting = df_voting[df_voting['Country name']=='England']
    df_deprivation['ConstituencyName'] = df_deprivation['ConstituencyName'].str.title()
    df_voting['Constituency name'] = df_voting['Constituency name'].str.title()
    df_m = pd.merge(df_deprivation,
                         df_voting,
                         how='left',
                         left_on='ConstituencyName',
                         right_on='Constituency name'
                        )
    df_m = pd.merge(df_m,
                         asmr,
                         how='left',
                         left_on='ONSConstID',
                         right_on='Parliamentary Constituency '
                        )
    df_m['Con PC'] = df_m['Con']/df_m['Valid votes']
    df_m['Lab PC'] = df_m['Lab']/df_m['Valid votes']
    df_m['Lib PC'] = df_m['LD'] / df_m['Valid votes']
    df_m['Brx PC'] = df_m['BRX'] / df_m['Valid votes']
    df_m['first_party_3'] = np.where((df_m['First party']=='Lab') |
                                          (df_m['First party']=='Con'),
                                           df_m['First party'],
                                          'Other')
    df_nhs = pd.read_csv(os.path.join(os.getcwd(),
                                      '..',
                                      'data',
                                      'raw' ,
                                      'nhs_digital',
                                      'health_conditions_constituency.csv'),
                         index_col=0
                        )
    df_nhs_HT = df_nhs[df_nhs['condition']=='High blood pressure (hypertension)']
    df_m = pd.merge(df_m,
                    df_nhs_HT[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code'
                    )
    df_m = df_m.rename({'prevalence%': 'prevalence%_HT'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    df_nhs_ST = df_nhs[df_nhs['condition']=='Stroke or transient ischaemic attack']
    df_m = pd.merge(df_m,
                    df_nhs_ST[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_ST'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    df_nhs_Asthma = df_nhs[df_nhs['condition']=='Asthma']
    df_m = pd.merge(df_m,
                    df_nhs_Asthma[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_Asthma'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    df_nhs_Depression = df_nhs[df_nhs['condition']=='Depression']
    df_m = pd.merge(df_m,
                    df_nhs_Depression[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_Depression'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    df_nhs_Epilepsy = df_nhs[df_nhs['condition']=='Epilepsy']
    df_m = pd.merge(df_m,
                    df_nhs_Epilepsy[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_Epilepsy'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    df_nhs_Diabetes = df_nhs[df_nhs['condition'] == 'Diabetes']
    df_m = pd.merge(df_m,
                    df_nhs_Diabetes[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_Diabetes'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    df_nhs_LD = df_nhs[df_nhs['condition'] == 'Learning disabilities']
    df_m = pd.merge(df_m,
                    df_nhs_LD[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_LD'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)


    df_nhs_Obesity = df_nhs[df_nhs['condition'] == 'Obesity']
    df_m = pd.merge(df_m,
                    df_nhs_Obesity[['pcon_code', 'prevalence%']],
                    how='left',
                    left_on='ONSConstID',
                    right_on='pcon_code')
    df_m = df_m.rename({'prevalence%': 'prevalence%_Obesity'}, axis=1)
    df_m = df_m.drop('pcon_code', axis=1)

    return df_m


def make_temporal_printouts(df_sii, df_hid):
    df_sii_f = df_sii[df_sii['sex']=='Females']
    print(f"In 2001-2003, the SII for females was: {df_sii_f.loc['2001-03','value']}")
    print(f"In 2018-2020, the SII for females was: {df_sii_f.loc['2018-20','value']}")

    df_sii_m = df_sii[df_sii['sex']=='Males']
    print(f"In 2001-2003, the SII for males was: {df_sii_m.loc['2001-03','value']}")
    print(f"In 2018-2020, the SII for males was: {df_sii_m.loc['2018-20','value']}")

    df_hid_m_10 = df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Male') &
                         (df_hid['Category'] == '10 = least deprived')]
    df_hid_m_1 = df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Male') &
                        (df_hid['Category'] == '01 = most deprived')]
    print(f"In 2011-2013, Male Healthy LE for 1st decile was: {df_hid_m_1[['Value']].loc['2011-13', 'Value']}")
    print(f"In 2011-2013, Male Healthy LE for 10th decile was: {df_hid_m_10[['Value']].loc['2011-13', 'Value']}")
    print(f"In 2018-2020, Male Healthy LE for 1st decile was: {df_hid_m_1[['Value']].loc['2018-20', 'Value']}")
    print(f"In 2018-2020, Male Healthy LE for 10th decile was: {df_hid_m_10[['Value']].loc['2018-20', 'Value']}")

    df_hid_f_10 = df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Female') &
                         (df_hid['Category'] == '10 = least deprived')]
    df_hid_f_1 = df_hid[(df_hid['Indicator'] == 'Healthy life expectancy at birth - Female') &
                        (df_hid['Category'] == '01 = most deprived')]
    print(f"In 2011-2013, Female Healthy LE for 1st decile was: {df_hid_f_1[['Value']].loc['2011-13', 'Value']}")
    print(f"In 2011-2013, Female Healthy LE for 10th decile was: {df_hid_f_10[['Value']].loc['2011-13', 'Value']}")
    print(f"In 2018-2020, Female Healthy LE for 1st decile was: {df_hid_f_1[['Value']].loc['2018-20', 'Value']}")
    print(f"In 2018-2020, Female Healthy LE for 10th decile was: {df_hid_f_10[['Value']].loc['2018-20', 'Value']}")