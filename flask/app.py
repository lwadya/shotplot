import os
import sys
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

root_dir = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(root_dir, 'code'))
from target_reader import target_reader

app = Flask(__name__)

# Default settings
img_dir = os.path.join(root_dir, 'flask/static/img/temp')
trg_size = 400
shot_size = trg_size / 100
min_hmap_value = .9
max_hmap_value = .25
coord_bins = 11

# Recommendation strings
reco_op = ('Your optimized score is significantly higher than your actual '
           'score, which means you are grouping better than your score '
           'indicates. Adjusting your sights may help you shoot better.')
reco_grp_base = 'Your %s grouping %s'
reco_grp_hori = 'horizontal'
reco_grp_vert = 'vertical'
reco_grp_both_1 = 'is great! Keep up the good work!'
reco_grp_hori_2 = ('is decent. Make sure your bow hand is steady and that '
                   'you\'re pulling your release straight back and aren\'t '
                   'plucking the bowstring.')
reco_grp_hori_3 = ('needs work. Make sure you\'re holding your bow steadily '
                   'through your shot. As you release you should be pulling '
                   'your hand straight back across the side of your face.')
reco_grp_vert_2 = ('is all right. Make sure your bow hand is steady, your '
                   'anchor point is consistent, and that you\'re reaching full '
                   'draw before releasing.')
reco_grp_vert_3 = ('could be better. Make sure you\'re holding your bow '
                   'steadily through your shot and aren\'t catching it as you '
                   'release. You should also work on finding a consistent '
                   'anchor point in order to improve your aim.')

# Recommendation thresholds
reco_op_pct = 1.1
reco_grp_std = [.1, .15, 1]

# Instantiates target reader
tr = target_reader()
score_step = trg_size * tr.score_step / tr.out_width

@app.route("/", methods=["POST", "GET"])
def homepage():
    if request.method == 'POST':
        img = request.files['filename']
        filename = img.filename
        if filename:
            filename = secure_filename(filename)
            fullpath = os.path.join(img_dir, filename)
            img.save(fullpath)
            error = tr.run(fullpath)
        else:
            error = 'No filename specified'

        if error:
            shots = stats = None
        else:
            # Corrects coordinates for display dimensions
            df = tr.df.copy(deep=True)
            for c in ['x', 'y', 'error', 'op_x', 'op_y']:
                df[c] *= trg_size / tr.out_width
            shots = df.to_dict(orient='records')

            # Calculates summary statistics
            num_shots = df.shape[0]
            mean_score = round(df['score'].mean(), 2)
            mean_op_score = round(df['op_score'].mean(), 2)
            mean_error = df['error'].mean()
            reco_list = get_reco_list(df['x'],
                                      df['y'],
                                      mean_score,
                                      mean_op_score)

            # Calculates values for score and coordinate heatmaps
            score_hist, score_hmap = series_to_hist(df['score'], 0, 10)
            op_score_hist, op_score_hmap = series_to_hist(df['op_score'], 0, 10)
            x_hist, x_hmap = coords_to_hist(df['x'])
            y_hist, y_hmap = coords_to_hist(df['y'])

            # Packages summary statistics into a dict
            stats = {
                'trg_size': trg_size,
                'shot_size': shot_size,
                'score_step': score_step,
                'num_shots': num_shots,
                'mean_score': mean_score,
                'mean_op_score': mean_op_score,
                'mean_error': mean_error,
                'recos': reco_list,
                'score_hist': score_hist,
                'score_hmap': score_hmap,
                'op_score_hist': op_score_hist,
                'op_score_hmap': op_score_hmap,
                'x_hist': x_hist,
                'x_hmap': x_hmap,
                'y_hist': y_hist,
                'y_hmap': y_hmap
            }

    else:
        error = shots = stats = filename = None

    return render_template('index.html',
                           error=error,
                           shots=shots,
                           stats=stats,
                           filename=filename)

# series_to_hist function, derives a histogram and heatmap values from a series
def series_to_hist(series, rng_min, rng_max):
    as_dict = series.value_counts().to_dict()
    hist = [as_dict.get(x, 0) for x in range(rng_min, rng_max+1)]
    hmap = np.interp(hist,
                     (0, max(hist)),
                     (min_hmap_value, max_hmap_value))
    return hist, hmap.tolist()

# coords_to_hist function, derives a histogram and heatmap values from a list of
# coordinates
def coords_to_hist(coords):
    bins = np.linspace(0, trg_size, coord_bins, endpoint=False)[1:]
    bin_vals = np.searchsorted(bins, coords)
    hist = np.bincount(bin_vals, minlength=11)
    hmap = np.interp(hist,
                     (0, max(hist)),
                     (min_hmap_value, max_hmap_value))
    return hist.tolist(), hmap.tolist()

# get_reco_list function, combines a list of recommendations based on shooting
# statistics
def get_reco_list(x_vals, y_vals, actual_score, op_score):
    reco_grp_hori_all = [reco_grp_both_1, reco_grp_hori_2, reco_grp_hori_3]
    reco_grp_vert_all = [reco_grp_both_1, reco_grp_vert_2, reco_grp_vert_3]
    reco_list = []

    hori_std = x_vals.std()
    hori_idx = np.searchsorted(reco_grp_std, hori_std / trg_size)
    vert_std = y_vals.std()
    vert_idx = np.searchsorted(reco_grp_std, vert_std / trg_size)
    hori_reco = reco_grp_base % (reco_grp_hori, reco_grp_hori_all[hori_idx])
    vert_reco = reco_grp_base % (reco_grp_vert, reco_grp_vert_all[vert_idx])

    if op_score > actual_score * reco_op_pct:
        reco_list.append(reco_op)
    if hori_std > vert_std:
        reco_list.extend([hori_reco, vert_reco])
    else:
        reco_list.extend([vert_reco, hori_reco])
    return reco_list

if __name__ == '__main__':
    app.run()
