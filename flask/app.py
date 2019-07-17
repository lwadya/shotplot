import os
import sys
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

root_dir = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(root_dir, 'code'))
from target_reader import target_reader
from shot_stats import shot_stats


# Sets app config values
app = Flask(__name__)
app.config['ALLOWED_IMAGE_EXTENSIONS'] =\
    ['BMP', 'DIB', 'JPEG', 'JPG', 'JPE', 'JP2', 'PNG', 'WEBP', 'TIF', 'TIFF']
app.config['MAX_IMAGE_FILESIZE'] = 5 * 1024 * 1024

# Default image settings
img_rel = 'static/sample_targets'
img_dir = os.path.join(root_dir, 'flask', img_rel)
trg_size = 400

# Instantiates target reader and shot stats objects
tr = target_reader()
ss = shot_stats(trg_size, tr.score_step, tr.out_width)


@app.route("/", methods=["POST", "GET"])
def index():
    '''
    Gathers input from user and displays results using a Flask HTML template

    Args:
        None

    Returns:
        None
    '''
    if request.method == 'POST':
        # First checks if a sample image was selected
        sample_img = request.form.get('sample', None)
        if sample_img:
            filename = os.path.basename(sample_img)
            img = os.path.join(img_dir, filename)
            error = tr.run(img, None)
        # Otherwise assumes an image file was uploaded
        else:
            img = request.files['filename']
            error, filename = check_upload(img)
            if not error:
                error = tr.run(None, img)

        # Throws error message if image processing failed
        if error:
            shots = stats = None
        # Otherwise attempts to calculate shot stats
        else:
            error = ss.run(tr.df)
            shots = ss.shots
            stats = ss.stats

    else:
        error = shots = stats = filename = None

    return render_template('index.html',
                           error=error,
                           shots=shots,
                           stats=stats,
                           filename=filename)

@app.route("/samples")
def samples():
    '''
    Displays sample target images page that can be used to launch visualizations

    Args:
        None

    Returns:
        None
    '''
    sample_imgs = os.listdir(img_dir)
    if sample_imgs:
        sample_imgs.sort()
        sample_imgs = [os.path.join('/', img_rel, i) for i in sample_imgs]
    else:
        sample_imgs = []

    return render_template('samples.html', images=sample_imgs)

def check_upload(file_obj):
    '''
    Checks user upload to make sure it meets file requirements

    Args:
        file_obj (werkzeug file object): user uploaded file

    Returns:
        None if successful, str containing error message if not
    '''
    # Makes sure filename exists
    filename = file_obj.filename
    if filename:
        filename = secure_filename(filename)
    else:
        return 'Filename not specified', None

    # Makes sure filename has any extension
    _, ext = os.path.splitext(filename)
    if not ext:
        return 'Filename has no extension', filename

    # Makes sure filename has a recognized extension
    ext = ext.replace('.', '').upper()
    if ext not in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        ext_list = ', '.join(app.config["ALLOWED_IMAGE_EXTENSIONS"])
        return f'File must be one of the following: {ext_list}', filename

    # Attempts to determine filesize
    try:
        file_obj.seek(0, os.SEEK_END)
        file_length = file_obj.tell()
        file_obj.seek(0, os.SEEK_SET)
    except:
        return 'Could not determine file size', filename

    # Makes sure file is under maximum size limit
    if file_length > app.config["MAX_IMAGE_FILESIZE"]:
        return 'File exceeds maximum size limit', filename

    # If all tests pass, assumes file is ok
    return None, filename


if __name__ == '__main__':
    app.run()
