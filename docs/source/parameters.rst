
Parameters:
-----------

debug: bool : Whether to return debug messages.
chans_order_{1, 15, 30} : Channel / plane reordering to put in order of tissue depth.
save_output : If true, will save each plane in a separate folder, otherwise saves the full volume
raw_data_dirs : Absolute path to the folder containing your data. Must be a list of 1 or more directories.
fname_must_contain : Optional string to match filenames to exclude from the analysis.
fname_must_NOT_contain: Optional string to match filenames to exclude from the analysis.
make_template_seams_and_plane_alignment : Flag to start reconstruction.
reconstruct_all_files : Whether to iterate over all files
reconstruct_until_this_ifile : Iterate over this many files in each raw_data_dirs. Fallback
list_files_for_template : TBD.
seams_overlap : "calculate",
make_nonan_volume :?? Isn't user set, checks for nan's in each plane
    - False if letaral_align_planes = True because it is going to be no-nan by definition, no need to check for it
lateral_align_planes : Check if  the pipeline can work with int16, and do it if so. If NaN
    handling is required, float32 will be used instead.
add_1000_for_nonegative_volume : Correct for the first 1000 planes being used by the resonant scanner?
save_mp4 : Saves the plane across time,
save_meanf_png : Saves the image of an individual plane.
