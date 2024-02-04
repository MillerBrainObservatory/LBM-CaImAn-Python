import numpy as np


def init_params():
    params = {}

    # TODO: detect the number of planes based on the file metadata and not on the filename
    params["chans_order_1plane"] = np.array([0])
    params["chans_order_15planes"] = (
        np.array([1, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 15]) - 1
    )
    params["chans_order_30planes"] = (
        np.array(
            [
                1,
                5,
                6,
                7,
                8,
                9,
                2,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                3,
                18,
                19,
                20,
                21,
                22,
                23,
                4,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
            ]
        )
        - 1
    )
    params["flynn_temp_param"] = True
    params["raw_data_dirs"] = [
        r"/v-data4/foconnell/data/lbm/"
    ]  # Must be a list with 1 or more dirs
    params[
        "fname_must_contain"
    ] = ""  # something you want to specify and that the desired filenames should contain
    params[
        "fname_must_NOT_contain"
    ] = "some_random_stuff"  # if not needed, leave something you know it is not in the filename
    params["make_template_seams_and_plane_alignment"] = True
    params["reconstruct_all_files"] = True
    params["list_files_for_template"] = [0]
    params["json_logging"] = False
    params[
        "seams_overlap"
    ] = "calculate"  # Should be either 'calculate', an integer, or a list of integers with length=n_planes
    params["save_output"] = True
    params[
        "make_nonan_volume"
    ] = True  # Whether to trim the edges so the output does not have nans. Also affects output as planes if lateral_aligned_planes==True (to compensate for X-Y shifts of MAxiMuM) or params['identical_mroi_overlaps_across_planes']==False (if seams from different planes are merged differently, then some planes will end up being larger than others)
    params[
        "lateral_align_planes"
    ] = False  # Calculates and compensates the X-Y of MAxiMuM
    params["add_1000_for_nonegative_volume"] = True
    params["save_mp4"] = True
    params["save_meanf_png"] = True

    # TODO: detect the number of planes based on the file metadata and not on the filename
    params["chans_order_1plane"] = np.array([0])
    params["chans_order_15planes"] = (
        np.array([1, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13, 14, 15]) - 1
    )
    params["chans_order_30planes"] = (
        np.array(
            [
                1,
                5,
                6,
                7,
                8,
                9,
                2,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                3,
                18,
                19,
                20,
                21,
                22,
                23,
                4,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
            ]
        )
        - 1
    )
    return params
