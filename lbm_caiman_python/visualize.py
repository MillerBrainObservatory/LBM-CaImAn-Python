import random
from pathlib import Path
from typing import Any as ArrayLike

import matplotlib as mpl
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, patches as patches, patheffects as path_effects

from lbm_caiman_python import calculate_centers
from lbm_caiman_python.util.signal import smooth_data


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
    >>> metrics_df = lbm_caiman_python.summary.metrics_df_from_files(metrics_files)
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
    >>> metrics_df = lcp.metrics_df_from_files(metrics_files)
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


def display_components(estimates, dims, num_random=5):
    """
    Display side-by-side plots of accepted and rejected components,
    and a separate figure with randomly selected components.

    Parameters
    ----------
    estimates : object
        Object containing spatial (A) and temporal (C) components and indices for accepted and rejected components.
    dims : tuple
        Dimensions of the field of view (FOV).
    num_random : int, optional
        Number of random components to display. Default is 5.
    """
    # Ensure idx_components and idx_components_bad exist
    if not hasattr(estimates, 'idx_components') or not hasattr(estimates, 'idx_components_bad'):
        raise ValueError("Estimates object must have 'idx_components' and 'idx_components_bad' attributes.")

    # Extract indices for accepted and rejected components
    idx_accepted = estimates.idx_components
    idx_rejected = estimates.idx_components_bad

    # Spatial components
    A = estimates.A

    # Temporal components
    C = estimates.C

    # Plot accepted components
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idx_accepted[:min(5, len(idx_accepted))]):
        plt.subplot(2, 5, i + 1)
        component_image = np.reshape(A[:, idx].toarray(), dims, order='F')
        plt.imshow(component_image, cmap='gray')
        plt.title(f"Accepted {i + 1}")
        plt.axis('off')

    for i, idx in enumerate(idx_accepted[:min(5, len(idx_accepted))]):
        plt.subplot(2, 5, i + 6)
        plt.plot(C[idx])
        plt.title(f"Trace {i + 1}")

    plt.suptitle("Accepted Components")
    plt.tight_layout()

    # Plot rejected components
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idx_rejected[:min(5, len(idx_rejected))]):
        plt.subplot(2, 5, i + 1)
        component_image = np.reshape(A[:, idx].toarray(), dims, order='F')
        plt.imshow(component_image, cmap='gray')
        plt.title(f"Rejected {i + 1}")
        plt.axis('off')

    for i, idx in enumerate(idx_rejected[:min(5, len(idx_rejected))]):
        plt.subplot(2, 5, i + 6)
        plt.plot(C[idx])
        plt.title(f"Trace {i + 1}")

    plt.suptitle("Rejected Components")
    plt.tight_layout()

    # Randomly selected components
    all_indices = list(range(A.shape[1]))
    random_indices = random.sample(all_indices, min(num_random, len(all_indices)))

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, num_random, i + 1)
        component_image = np.reshape(A[:, idx].toarray(), dims, order='F')
        plt.imshow(component_image, cmap='gray')
        plt.title(f"Random {i + 1}")
        plt.axis('off')

    for i, idx in enumerate(random_indices):
        plt.subplot(2, num_random, i + num_random + 1)
        plt.plot(C[idx])
        plt.title(f"Trace {i + 1}")

    plt.suptitle("Random Components")
    plt.tight_layout()

    # Show all plots
    plt.show()


def plot_spatial_components(data: pd.DataFrame | pd.Series, savepath: str | Path | None = None, marker_size=3):
    """
    Plot spatial CNMF components for a DataFrame or a Series.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        A DataFrame containing CNMF data or a single Series (row) from the DataFrame.
    savepath : str, Path, or None, optional
        Directory to save the plots. If None, plots are not saved. Default is None.
    marker_size : int, optional
        Size of the markers for the center points. Set to 0 to skip drawing centers. Default is 3.

    Returns
    -------
    None
        Displays the plots and optionally saves them to the specified directory.

    Notes
    -----
    - The function handles both `pandas.DataFrame` and `pandas.Series` as input.
    - If `marker_size` is set to 0, no center points are drawn on the plot.
    - The `savepath` must be a valid directory path if saving is enabled.

    Examples
    --------
    For a DataFrame:
    >>> plot_spatial_components(df, savepath="./plots", marker_size=5)

    For a single row (Series):
    >>> plot_spatial_components(df.iloc[0], savepath="./plots", marker_size=5)
    """
    if isinstance(data, pd.DataFrame):
        for idx, row in data.iterrows():
            if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
                print(f"Skipping {row.uuid} as it is not successful.")
                continue

            if row["algo"] == "cnmf":
                model = row.cnmf.get_output()
                red_idx = model.estimates.idx_components_bad

                spatial_footprints = model.estimates.A
                dims = (model.dims[1], model.dims[0])

                max_proj = spatial_footprints.max(axis=1).toarray().reshape(dims)
                plt.imshow(max_proj, cmap="gray")

                # Check marker size
                if marker_size == 0:
                    print('Skipping drawing centers')
                else:
                    print(f'Marker size is set to {marker_size}')
                    centers = calculate_centers(spatial_footprints, dims)
                    colors = ['b'] * len(centers)

                    for i in red_idx:
                        colors[i] = 'r'
                    plt.scatter(centers[:, 0], centers[:, 1], c=colors, s=marker_size, marker='.')

                plt.tight_layout()
                plt.show()
                if savepath:
                    save_name = Path(savepath) / f"{row.uuid}_spatial_components.png"
                    print(f"Saving to {save_name}.")
                    plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")
    else:
        row = data
        if isinstance(row["outputs"], dict) and not row["outputs"].get("success") or row["outputs"] is None:
            print(f"Skipping {row.uuid} as it is not successful.")
            return

        if row["algo"] == "cnmf":
            model = row.cnmf.get_output()
            red_idx = model.estimates.idx_components_bad

            spatial_footprints = model.estimates.A
            dims = (model.dims[1], model.dims[0])

            max_proj = spatial_footprints.max(axis=1).toarray().reshape(dims)
            plt.imshow(max_proj, cmap="gray")

            # Check marker size
            if marker_size == 0:
                print('Skipping drawing centers')
            else:
                print(f'Marker size is set to {marker_size}')
                centers = calculate_centers(spatial_footprints, dims)
                colors = ['b'] * len(centers)

                for i in red_idx:
                    colors[i] = 'r'
                plt.scatter(centers[:, 0], centers[:, 1], c=colors, s=marker_size, marker='.')

            plt.tight_layout()
            plt.show()
            if savepath:
                save_name = Path(savepath) / f"{row.uuid}_spatial_components.png"
                print(f"Saving to {save_name}.")
                plt.savefig(save_name.expanduser(), dpi=600, bbox_inches="tight")
