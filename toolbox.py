

def get_pos_at_frame(df, frame_idx):
    """Get the (x,y) positions of all the targets existing for the given frame index"""
    targets = df[df['frame'] == frame_idx]
    pos_at_frame = targets[['x', 'y']].to_numpy().T
    return pos_at_frame
