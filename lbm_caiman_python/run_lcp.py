# mcorr_movie = df.iloc[0].mcorr.get_output()
# cnmf_model = df.iloc[-1].cnmf.get_output()
# contours = df.iloc[-1].cnmf.get_contours()
#
# good_masks = df.iloc[-1].cnmf.get_masks('good')
# bad_masks = df.iloc[-1].cnmf.get_masks('bad')
#
# combined_masks = np.argmax(good_masks, axis=-1) + 1  # +1 to avoid zero for the background
# all_masks = dask.array.stack([mask[..., i] for i, mask in enumerate(good_masks)])
# correlation_image = df.iloc[-1].caiman.get_corr_image()
