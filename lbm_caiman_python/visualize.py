from typing import Any as ArrayLike

import matplotlib as mpl
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, patches as patches, patheffects as path_effects
import fastplotlib as fpl

from lbm_caiman_python.util.signal import smooth_data

def export_contours_with_params(row, save_path):
    params = row.params
    corr = row.caiman.get_corr_image()
    contours = row.cnmf.get_contours("good", swap_dim=False)[0]
    contours_bad = row.cnmf.get_contours("bad", swap_dim=False)[0]

    table_data = params["main"]
    df_table = pd.DataFrame(list(table_data.items()), columns=["Parameter", "Value"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(corr, cmap='gray')
    for contour in contours:
        axes[0].plot(contour[:, 0], contour[:, 1], color='cyan', linewidth=1)
    for contour in contours_bad:
        axes[0].plot(contour[:, 0], contour[:, 1], color='red', linewidth=0.2)

    axes[0].set_title(f'Accepted ({len(contours)}) and Rejected ({len(contours_bad)}) Neurons')
    axes[0].axis('off')
    axes[1].axis('tight')
    axes[1].axis('off')

    table = axes[1].table(cellText=df_table.values,
                          colLabels=df_table.columns,
                          loc='center',
                          cellLoc='center',
                          colWidths=[0.4, 0.6])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_contours(df, plot_index, histogram_widget=False):
    """
    Plot the contours of the accepted and rejected components.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the CNMF pandas extension.
    plot_index : int
        Index of the DataFrame to plot.
    histogram_widget : bool, optional
        Flag to display the vmin/vmax histogram controller.

    Returns
    -------
    fpl.ImageWidget
        Widget with 2 subplots containing the contours of the accepted and rejected components.
    """
    model = df.iloc[plot_index].cnmf.get_output()
    print(f"Accepted: {len(model.estimates.idx_components)} | Rejected: {len(model.estimates.idx_components_bad)}")
    contours_g = df.iloc[plot_index].cnmf.get_contours("good", swap_dim=False)
    contours_b = df.iloc[plot_index].cnmf.get_contours("bad", swap_dim=False)
    mcorr_movie = df.iloc[plot_index].caiman.get_input_movie()

    image_widget = fpl.ImageWidget(
        data=[mcorr_movie, mcorr_movie],
        names=['Accepted', 'Rejected'],
        window_funcs={'t': (np.mean, 3)},
        figure_kwargs={'size': (1200, 600)},
        histogram_widget=histogram_widget
        figure_kwargs={'size': (1200, 600)}
    )
    for subplot in image_widget.figure:
        if subplot.name == 'Accepted':
            subplot.add_line_collection(
                contours_g[0],
                name="contours"
            )
        elif subplot.name == 'Rejected':
            subplot.add_line_collection(
                contours_b[0],
                name="contours"
            )
    return image_widget


def export_contours_with_params(row, save_path):
    params = row.params
    corr = row.caiman.get_corr_image()
    contours = row.cnmf.get_contours("good", swap_dim=False)[0]
    contours_bad = row.cnmf.get_contours("bad", swap_dim=False)[0]

    table_data = params["main"]
    df_table = pd.DataFrame(list(table_data.items()), columns=["Parameter", "Value"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(corr, cmap='gray')
    for contour in contours:
        axes[0].plot(contour[:, 0], contour[:, 1], color='cyan', linewidth=1)
    for contour in contours_bad:
        axes[0].plot(contour[:, 0], contour[:, 1], color='red', linewidth=0.2)

    axes[0].set_title(f'Accepted ({len(contours)}) and Rejected ({len(contours_bad)}) Neurons')
    axes[0].axis('off')
    axes[1].axis('tight')
    axes[1].axis('off')

    table = axes[1].table(cellText=df_table.values,
                          colLabels=df_table.columns,
                          loc='center',
                          cellLoc='center',
                          colWidths=[0.4, 0.6])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_contours(df, plot_index, histogram_widget=False):
    """
    Plot the contours of the accepted and rejected components.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the CNMF pandas extension.
    plot_index : int
        Index of the DataFrame to plot.
    histogram_widget : bool, optional
        Flag to display the vmin/vmax histogram controller.

    Returns
    -------
    fpl.ImageWidget
        Widget with 2 subplots containing the contours of the accepted and rejected components.
    """
    model = df.iloc[plot_index].cnmf.get_output()
    print(f"Accepted: {len(model.estimates.idx_components)} | Rejected: {len(model.estimates.idx_components_bad)}")
    contours_g = df.iloc[plot_index].cnmf.get_contours("good", swap_dim=False)
    contours_b = df.iloc[plot_index].cnmf.get_contours("bad", swap_dim=False)
    mcorr_movie = df.iloc[plot_index].caiman.get_input_movie()

    image_widget = fpl.ImageWidget(
        data=[mcorr_movie, mcorr_movie],
        names=['Accepted', 'Rejected'],
        window_funcs={'t': (np.mean, 3)},
        figure_kwargs={'size': (1200, 600)},
        histogram_widget=histogram_widget
    )
    for subplot in image_widget.figure:
        if subplot.name == 'Accepted':
            subplot.add_line_collection(
                contours_g[0],
                name="contours"
            )
        elif subplot.name == 'Rejected':
            subplot.add_line_collection(
                contours_b[0],
                name="contours"
            )
    return image_widget


def plot_with_scalebars(image: ArrayLike, pixel_resolution: float):
    """
    Plot a 2D image with scale bars of 5, 10, and 20 microns.

    Parameters
    ----------
    image : ndarray
        A 2D NumPy array representing the image to be plotted.
    pixel_resolution : float
        The resolution of the image in microns per pixel.

    Returns
    -------
    None
    """
    scale_bar_sizes = [5, 10, 20]  # Sizes of scale bars in microns

    # Calculate the size of scale bars in pixels for each bar size
    scale_bar_lengths = [int(size / pixel_resolution) for size in scale_bar_sizes]

    # Create subplots to display each version of the image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, scale_length, size in zip(axes, scale_bar_lengths, scale_bar_sizes):
        ax.imshow(image, cmap='gray')

        # Determine image dimensions for dynamic placement of scale bar
        image_height, image_width = image.shape

        # Scale bar thickness is 1% of the image height, but at least 2px thick
        bar_thickness = max(2, int(0.01 * image_height))  # Thinner bar than before

        # Center the scale bar horizontally and vertically
        bar_x = (image_width // 2) - (scale_length // 2)  # Centered horizontally
        bar_y = (image_height // 2) - (bar_thickness // 2)  # Centered vertically

        # Draw the scale bar
        ax.add_patch(patches.Rectangle((bar_x, bar_y), scale_length, bar_thickness,
                                       color='white', edgecolor='black', linewidth=1))

        # Add annotation for the scale bar (below the bar)
        font_size = max(10, int(0.03 * image_height))  # Font size relative to image size
        text = ax.text(bar_x + scale_length / 2, bar_y + bar_thickness + font_size + 5,
                       f'{size} μm', color='white', ha='center', va='top',
                       fontsize=font_size, fontweight='bold')

        # Apply a stroke effect to the text for better contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

        # Remove axis for a clean image
        ax.axis('off')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()


def plot_optical_flows(input_df: pd.DataFrame, max_columns=4, save_path=None):
    """
    Plots the dense optical flow images from a DataFrame containing metrics information.

    Parameters
    ----------
    input_df : DataFrame
        DataFrame containing 'flows', 'batch_index', 'mean_corr', 'mean_norm', 'crispness', and other related columns.
        Typically, use the output of `create_metrics_df` to get the input DataFrame.
    max_columns : int, optional
        Maximum number of columns to display in the plot. Default is 4.

    Examples
    --------
    >>> import lbm_caiman_python as lcp
    >>> import lbm_mc as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> metrics_files = lbm_caiman_python.summary.compute_mcorr_metrics_batch(batch_df)
    >>> metrics_df = lbm_caiman_python.summary._create_df_from_metric_files(metrics_files)
    >>> lcp.plot_optical_flows(metrics_df, max_columns=2)
    """
    num_graphs = len(input_df)
    num_rows = int(np.ceil(num_graphs / max_columns))

    fig, axes = plt.subplots(num_rows, max_columns, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    flow_images = []

    highest_corr_batch = input_df.loc[input_df['mean_corr'].idxmax()]['batch_index']
    highest_crisp_batch = input_df.loc[input_df['crispness'].idxmax()]['batch_index']
    lowest_norm_batch = input_df.loc[input_df['mean_norm'].idxmin()]['batch_index']

    for i, (index, row) in enumerate(input_df.iterrows()):
        # Avoid indexing beyond available axes if there are more df than plots
        if i >= len(axes):
            break
        ax = axes[i]

        batch_idx = row['batch_index']
        metric_path = row['metric_path']
        with np.load(metric_path) as f:
            flows = f['flows']
            flow_img = np.mean(np.sqrt(flows[:, :, :, 0] ** 2 + flows[:, :, :, 1] ** 2), axis=0)
            del flows  # free up expensive array
            flow_images.append(flow_img)

        ax.imshow(flow_img, vmin=0, vmax=0.3, cmap='viridis')

        title_parts = []

        # Title Part 1: Item and Batch Index
        if batch_idx == -1:
            item_title = "Raw Data"
        else:
            item_title = f'Batch Index: {batch_idx}'

        if batch_idx == highest_corr_batch:
            item_title = f'Batch Index: {batch_idx} **(Highest Correlation)**'
        title_parts.append(item_title)

        mean_norm = row['mean_norm']
        norm_title = f'ROF: {mean_norm:.2f}'
        if batch_idx == lowest_norm_batch:
            norm_title = f'ROF: **{mean_norm:.2f}** (Lowest Norm)'
        title_parts.append(norm_title)

        smoothness = row['crispness']
        crisp_title = f'Crispness: {smoothness:.2f}'
        if batch_idx == highest_crisp_batch:
            crisp_title = f'Crispness: **{smoothness:.2f}** (Highest Crispness)'
        title_parts.append(crisp_title)

        title = '\n'.join(title_parts)

        ax.set_title(
            title,
            fontsize=14,
            fontweight='bold',
            color='black',
            loc='center'
        )

        ax.axis('off')

    # Turn off unused axes
    for i in range(len(input_df), len(axes)):
        axes[i].axis('off')

    cbar_ax = fig.add_axes((0.92, 0.2, 0.02, 0.6))
    norm = mpl.colors.Normalize(vmin=0, vmax=0.3)
    sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    cbar.set_label('Flow Magnitude', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 0.9, 1))
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual flows saved to {save_path}")


def plot_residual_flows(results, num_batches=3, smooth=True, winsize=5, save_path=None):
    """
    Plot the top `num_batches` residual optical flows across batches.

    Parameters
    ----------
    results : DataFrame
        DataFrame containing 'uuid' and 'batch_index' columns.
    num_batches : int, optional
        Number of "best" batches to plot. Default is 3.
    smooth : bool, optional
        Whether to smooth the residual flows using a moving average. Default is True.
    winsize : int, optional
        The window size for smoothing the data. Default is 5.

    Examples
    --------
    >>> import lbm_caiman_python as lcp
    >>> import lbm_mc as mc
    >>> batch_df = mc.load_batch('path/to/batch.pickle')
    >>> metrics_files = lcp.compute_mcorr_metrics_batch(batch_df)
    >>> metrics_df = lcp._create_df_from_metric_files(metrics_files)
    >>> lcp.plot_residual_flows(metrics_df, num_batches=6, smooth=True, winsize=8)
    """
    # Sort and filter for top batches by mean_norm, lower is better
    results_sorted = results.sort_values(by='mean_norm')
    top_uuids = results_sorted['uuid'].values[:num_batches]
    results_filtered = results[results['uuid'].isin(top_uuids)]

    # Identify raw data UUID
    raw_uuid = results.loc[results['uuid'].str.contains('raw', case=False, na=False), 'uuid'].values[0]
    best_uuid = top_uuids[0]  # Best (lowest) value

    fig, ax = plt.subplots(figsize=(20, 10))

    colors = plt.cm.Set1(np.linspace(0, 1, num_batches))  # Standout colors for other batches
    plotted_uuids = set()  # Track plotted UUIDs to avoid duplicates

    if raw_uuid in results['uuid'].values:
        row = results.loc[results['uuid'] == raw_uuid].iloc[0]
        metric_path = row['metric_path']

        with np.load(metric_path) as metric:
            flows = metric['flows']

        residual_flows = [np.linalg.norm(flows[i] - flows[i - 1], axis=2).mean() for i in range(1, len(flows))]

        if smooth:
            residual_flows = smooth_data(residual_flows, window_size=winsize)

        if raw_uuid == best_uuid:
            ax.plot(residual_flows, color='blue', linestyle='dotted', linewidth=2.5,
                    label=f'Best (Raw)')
        else:
            ax.plot(residual_flows, color='red', linestyle='dotted', linewidth=2.5,
                    label=f'Raw Data')

        plotted_uuids.add(raw_uuid)  # Add raw UUID to avoid double plotting

    for i, row in results_filtered.iterrows():
        file_uuid = row['uuid']
        batch_idx = row['batch_index']

        if file_uuid in plotted_uuids:
            continue

        metric_path = row['metric_path']

        with np.load(metric_path) as metric:
            flows = metric['flows']

        residual_flows = [np.linalg.norm(flows[i] - flows[i - 1], axis=2).mean() for i in range(1, len(flows))]

        if smooth:
            residual_flows = smooth_data(residual_flows, window_size=winsize)

        if file_uuid == best_uuid:
            ax.plot(residual_flows, color='blue', linestyle='solid', linewidth=2.5,
                    label=f'Best Value | Batch Row Index: {batch_idx}')
        else:
            color_idx = list(top_uuids).index(file_uuid) if file_uuid in top_uuids else len(plotted_uuids) - 1
            ax.plot(residual_flows, color=colors[color_idx], linestyle='solid', linewidth=1.5,
                    label=f'Batch Row Index: {batch_idx}')

        plotted_uuids.add(file_uuid)

    ax.set_xlabel('Frames (downsampled)', fontsize=12, fontweight='bold')

    # Make X tick labels bold
    ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontweight='bold')

    # Make Y tick labels bold
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontweight='bold')

    ax.set_ylabel('Residual Optical Flow (ROF)', fontsize=12, fontweight='bold')
    ax.set_title(f'Batches with Lowest Residual Optical Flow', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, title='Figure Key', title_fontsize=12, prop={'weight': 'bold'})
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual flows saved to {save_path}")


def plot_correlations(results, num_batches=3, smooth=True, winsize=5, save_path=None):
    """
    Plot the top `num_batches` batches with the highest correlation coefficients.

    Parameters
    ----------
    results : DataFrame
        DataFrame containing 'uuid' and 'batch_index' columns.
    num_batches : int, optional
        Number of "best" batches to plot. Default is 3.
    smooth : bool, optional
        Whether to smooth the correlation data using a moving average. Default is True.
    winsize : int, optional
        The window size for smoothing the data. Default is 5.
    """
    results_sorted = results.sort_values(by='mean_corr', ascending=False)
    top_uuids = results_sorted['uuid'].values[:num_batches]
    results_filtered = results[results['uuid'].isin(top_uuids)]

    raw_uuid = results.loc[results['uuid'].str.contains('raw', case=False, na=False), 'uuid'].values[0]
    best_uuid = top_uuids[0]

    fig, ax = plt.subplots(figsize=(20, 10))

    colors = plt.cm.Set1(np.linspace(0, 1, num_batches))
    plotted_uuids = set()

    if raw_uuid in results['uuid'].values:
        row = results.loc[results['uuid'] == raw_uuid].iloc[0]
        metric_path = row['metric_path']

        with np.load(metric_path) as metric:
            correlations = metric['correlations']

        if smooth:
            correlations = smooth_data(correlations, window_size=winsize)

        if raw_uuid == best_uuid:
            ax.plot(correlations, color='blue', linestyle='dotted', linewidth=2.5,
                    label=f'Best (Raw)')
        else:
            ax.plot(correlations, color='red', linestyle='dotted', linewidth=2.5,
                    label=f'Raw Data')

        plotted_uuids.add(raw_uuid)

    for i, row in results_filtered.iterrows():
        file_uuid = row['uuid']
        batch_idx = row['batch_index']

        if file_uuid in plotted_uuids:
            continue

        metric_path = row['metric_path']

        with np.load(metric_path) as metric:
            correlations = metric['correlations']

        if smooth:
            correlations = smooth_data(correlations, window_size=winsize)

        if file_uuid == best_uuid:
            ax.plot(correlations, color='blue', linestyle='solid', linewidth=2.5,
                    label=f'Best Value | Batch Row Index: {batch_idx}')
        else:
            color_idx = list(top_uuids).index(file_uuid) if file_uuid in top_uuids else len(plotted_uuids) - 1
            ax.plot(correlations, color=colors[color_idx], linestyle='solid', linewidth=1.5,
                    label=f'Batch Row Index: {batch_idx}')

        plotted_uuids.add(file_uuid)

    ax.set_xlabel('Frame Index (Downsampled)', fontsize=12, fontweight='bold')

    ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontweight='bold')
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontweight='bold')
    ax.set_ylabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    ax.set_title(f'Batches with Highest Correlation', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, title='Figure Key', title_fontsize=12, prop={'weight': 'bold'})
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual flows saved to {save_path}")
