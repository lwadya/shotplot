# Flask imports
from flask import Flask
from flask import request, render_template

# Target Reader imports
import os
import numpy as np
from target_reader import target_reader

app = Flask(__name__)

# Default settings
img_dir = ('/Users/lukaswadya/dataScience/metis/metisgh/metis_work/05_kojak/'
           'img/app_test')
trg_size = 400
shot_size = trg_size / 100
min_hmap_value = .9
max_hmap_value = .3
coord_bins = 11

# Instantiates target reader
tr = target_reader()
score_step = trg_size * tr.score_step / tr.out_width

@app.route("/", methods=["POST", "GET"])
def homepage():
    if request.method == 'POST':
        filename = request.form.get('filename')
        filename = os.path.join(img_dir, filename)
        error = tr.run(filename)

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
                'mean_score': mean_score,
                'mean_op_score': mean_op_score,
                'mean_error': mean_error,
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

if __name__ == '__main__':
    app.run()
