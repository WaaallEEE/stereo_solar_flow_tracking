import os
from pathlib import PurePath
import pandas as pd
import numpy as np
from sunflower.balltracking import mballtrack as mblt
from stereo_flow_tracking import toolbox
from stereo_flow_tracking import config


if __name__ == '__main__':
    """Applies Magnetic Balltracking to get the optical flows on the STEREO COR 2 Data"""
    # TODO: Prepped data are generated in the COR2_tracking_prep.ipynb
    #  Must Convert the notebook into a proper python file.

    df_targets = pd.read_csv(config.TRAINING_SET_CSV, usecols=['frame', 'blob_id_init', 'x', 'y'])
    init_target_pos = toolbox.get_pos_at_frame(df_targets, 0)
    # Enable exporting figures of the balls' center overlayed on top of the original images
    print_figures = True
    # config dictionnary for the parameters that aren't subject to parameter sweeps
    mbt_fixed_params = config.mbt_dict
    # Ball params that can be subject to parameter sweep
    ball_params = {
        "rs": 4,
        "am": 1,
        "dp": 0.3,
        "tdx": 1,
        "tdy": 100,
        "zdamping": 1,
        "ballspacing": 15,
        "intsteps": 20,
        "mag_thresh": 3.5,
        "noise_level": 2
    }

    mbt = mblt.MBT(init_pos=init_target_pos, **ball_params, **mbt_fixed_params)
    mbt.track_all_frames()

    df_targets[['ball_x', 'ball_y']] = np.NaN
    for i in range(mbt_fixed_params['nt']):
        # New column to insert
        df_targets.loc[df_targets['frame'] == i, 'ball_x'] = mbt.ballpos[0, :, i]
        df_targets.loc[df_targets['frame'] == i, 'ball_y'] = mbt.ballpos[1, :, i]

    df_targets.to_csv(PurePath(config.OUTPUT_DIR, 'df_targets.csv'), index=False)

    # Export figures
    if print_figures:
        os.makedirs(config.FIG_DIR, exist_ok=True)
        mbt.export_track_figures(axlims=[0, 600, 0, 659], vmin=0, vmax=5, cmap='gray_r')



