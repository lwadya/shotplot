import numpy as np
import pandas as pd
import cv2


class target_reader:
    '''
    Reads in an image of a used archery target and uses openCV to determine
    position and score value for each shot. __init__ initializes session
    settings and run performs analysis.
    '''

    # Class-wide settings
    # Real-world target dimensions in cm
    cm_width = 42
    blue_ratio = 24 / cm_width
    # HSV ranges for colored regions of the target
    colors = {
        'yellow': [{'low': np.array([15, 130, 130]),
                    'high': np.array([45, 255, 255])}],
        'red': [{'low': np.array([165, 130, 130]),
                 'high': np.array([180, 255, 255])},
                {'low': np.array([0, 130, 130]),
                 'high': np.array([15, 255, 255])}],
        'blue': [{'low': np.array([80, 80, 80]),
                  'high': np.array([130, 255, 255])}],
        'black': [{'low': np.array([0, 0, 0]),
                   'high': np.array([180, 255, 130])}],
    }
    # Counting order of outer ring for each colored region
    color_steps = {
        'yellow': 2,
        'red': 4,
        'blue': 6,
        'black': 8
    }

    def __init__(self, out_width=600):
        '''
        Establishes the output width of the processed target images

        Args:
            out_width (int): width/height of processed images in pixels

        Returns:
            None
        '''
        self.out_width = out_width
        self.score_step = out_width * 2 / self.cm_width
        return None

    def run(self, filename):
        '''
        Runs all methods for image processing and scoring. Returns None if
        analysis is successful, error message if not. Results are saved in the
        class variable 'df' and image steps in 'stage_images.'

        Args:
            filename (str): filepath of the image to analyze

        Returns:
            None if successful, str containing error message if not
        '''
        # Resets class variables
        self.orig_image = None
        self.image = None
        self.image_gray = None
        self.stage_images = []
        self.keypoints = None
        self.df = None

        # Loads image from file if it exists
        image = cv2.imread(filename, 1)
        if type(image) == type(None):
            return 'Could not read image file'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.orig_image = image
        self.image = image.copy()

        # Runs each step in the process, any error returns a message
        if self.remove_skew():
            return 'Could not find target corners'
        if self.standardize_size():
            return 'Could not identify target'
        if self.balance_contrast():
            return 'Could not balance contrast'
        if self.find_shots():
            return 'Could not detect shots'
        if self.get_shot_data():
            return 'Could not create shot dataframe'

        return None

    def remove_skew(self):
        '''
        Unskews perspective by moving each target corner to the corner of a new
        image

        Args:
            None

        Returns:
            None if successful, True if not
        '''
        # Function settings
        gray_width = 600
        filter_d = gray_width // 20
        filter_sigma = gray_width // 15
        canny_t1 = 100
        canny_t2 = 400
        dilate_kernel = 4
        perim_pct = .015
        min_area_ratio = .3

        # Converts colorspaces and resize images for edge detection
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height = int(gray_width / img.shape[1] * img.shape[0])
        gray = cv2.resize(gray, (gray_width, height))

        # Finds edges of target using grayscale image
        gray = cv2.bilateralFilter(gray,
                                   filter_d,
                                   filter_sigma,
                                   filter_sigma)
        edges = cv2.Canny(gray, canny_t1, canny_t2)
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Finds the longest contour that can be approximated by four coordinates
        # Returns error status if no such contour exists
        _, contours, _ = cv2.findContours(edges,
                                          cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        paper = None
        max_perim = 0
        for c in contours:
            perim = cv2.arcLength(c, True)
            if perim > max_perim:
                approx = cv2.approxPolyDP(c, perim * perim_pct, True)
                if len(approx) == 4:
                    paper = approx
                    max_perim = perim
        if type(paper) == type(None):
            return True

        # Reorders corner points to Top Left, Top Right, Bottom Right,
        # Bottom Left
        paper = paper.reshape(4, 2)
        bounds = paper.copy()
        sums = np.sum(paper, axis=1)
        diffs = np.diff(paper, axis=1)
        bounds[0] = paper[np.argmin(sums)]
        bounds[1] = paper[np.argmin(diffs)]
        bounds[2] = paper[np.argmax(sums)]
        bounds[3] = paper[np.argmax(diffs)]

        # Corrects skew and crops color image to paper
        bounds = (bounds * (img.shape[1] / gray_width)).astype('float32')
        top_w = np.linalg.norm(bounds[0] - bounds[1])
        btm_w = np.linalg.norm(bounds[2] - bounds[3])
        lft_h = np.linalg.norm(bounds[0] - bounds[3])
        rgt_h = np.linalg.norm(bounds[1] - bounds[2])
        new_w = int(min(top_w, btm_w))
        new_h = int(min(lft_h, rgt_h))
        new_bounds = np.array([
            [0, 0],
            [new_w, 0],
            [new_w, new_h],
            [0, new_h]
        ], dtype = 'float32')
        M = cv2.getPerspectiveTransform(bounds, new_bounds)
        img = cv2.warpPerspective(img, M, (new_w, new_h))

        # Returns error status if resized image area is below minimum ratio
        new_area = np.prod(img.shape)
        orig_area = np.prod(self.image.shape)
        if (new_area / orig_area) < min_area_ratio:
            return True

        # Saves image to class variables and returns status
        self.image = img
        self.stage_images.append(img.copy())
        return None

    def standardize_size(self):
        '''
        Resizes image to fit the standard template - meaning image and inner
        target circle dimensions match preset values

        Args:
            None

        Returns:
            None if successful, True if not
        '''
        # Function settings
        keys = ['yellow', 'red', 'blue']
        close_kernel = 2
        max_center_dist = self.out_width * .05

        # Finds circle centers and blue circle width
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        circle_data = []
        circle_contours = []
        for key in keys:
            mask = 0
            for rng in self.colors[key]:
                lyr = cv2.inRange(hsv, rng['low'], rng['high'])
                mask = cv2.bitwise_or(lyr, mask)
            kernel = np.ones((close_kernel, close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Finds circle contour and contour center
            _, contours, __ = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(contour) for contour in contours]
            idx = np.argsort(areas)[-1]
            M = cv2.moments(contours[idx])
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            circle_contours.append(contours[idx].copy())
            circle_data.append([areas[idx], center_x, center_y])

        # Checks circle data to make sure areas get larger by circle and centers
        # roughly align, returns error status if not
        circle_data = np.array(circle_data)
        target_center = np.mean(circle_data[:, 1:], axis=0).astype(int)
        last_area = 0
        for c in circle_data:
            dist_to_mean = np.linalg.norm(c[1:] - target_center)
            if c[0] < last_area or dist_to_mean > max_center_dist:
                return True
            last_area = c[0]

        # Scales image to final output size
        x, y, w, h = cv2.boundingRect(circle_contours[-1])
        blue_dim = self.blue_ratio * self.out_width
        scl_x = blue_dim / w
        scl_y = blue_dim / h
        img = cv2.resize(self.image, None, fx=scl_x, fy=scl_y)

        # Adds border if it is necessary to match final output size
        border_size = self.out_width // 10
        img = cv2.copyMakeBorder(img, *([border_size] * 4), cv2.BORDER_REFLECT)
        st_x = int(target_center[0] * scl_x + border_size - self.out_width / 2)
        st_y = int(target_center[1] * scl_y + border_size - self.out_width / 2)
        img = img[st_y:st_y+self.out_width, st_x:st_x+self.out_width]

        # Saves image to class variables and returns status
        self.image = img
        self.stage_images.append(img.copy())
        return None

    def balance_contrast(self):
        '''
        Adjusts image values to make arrow holes easier to detect against each
        different background color

        Args:
            None

        Returns:
            None if successful, True if not
        '''
        # Function settings
        keys = ['red', 'blue', 'black']
        pct_max = .01
        filter_d = self.out_width // 40
        filter_sigma = self.out_width // 30
        morph_kernel = 2
        clahe_limit = 1
        clahe_grid = 12
        logos = np.array([
            [[5, 525], [55, 600]],
            [[465, 545], [535, 600]],
            [[545, 525], [595, 600]]
        ])
        ref_size = 600

        # Convert to grayscale using HSV because it better separates values
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        gray = hsv[:, :, 2]

        # Creates masks for each area of the target with a different background
        circle_masks = []
        for key in keys:
            target_area = (np.square(self.color_steps[key] * self.score_step) *
                           np.pi)
            circle = 0
            for rng in self.colors[key]:
                circle = cv2.bitwise_or(cv2.inRange(hsv, rng['low'],
                                                    rng['high']), circle)

            # Finds contour that best matches estimated circle area
            _, contours, __ = cv2.findContours(circle,
                                               cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            idx = np.argmin(np.abs(areas - target_area))
            contour = contours[idx]

            # Creates mask based on contour
            mask = np.zeros_like(gray)
            mask = cv2.fillPoly(mask, pts=[contour], color=255)
            circle_masks.append(mask)

        # Removes smaller circle areas from larger ones, like a hole to a donut
        circle_masks.append(np.full_like(gray, 255))
        for x in range(len(circle_masks)-1, 0, -1):
            circle_masks[x] -= circle_masks[x-1]

        # Rebalances values based on histogram of each masked area
        layers = 0
        for mask in circle_masks:
            # Creates histogram and extracts most frequent value bin
            hist = cv2.calcHist([gray], [0], mask, [32], [0,256])
            mode_idx = np.argmax(hist)
            limit = hist[mode_idx] * pct_max

            # Calculates upper and lower value bounds
            limit_idx = np.argwhere(hist < limit)[:,0]
            try:
                low_val = max(limit_idx[limit_idx < mode_idx]) * 8
            except:
                low_val = 0
            try:
                high_val = min(limit_idx[limit_idx > mode_idx]) * 8
            except:
                high_val = 255

            # Creates a new image on which to perform rebalancing
            layer = gray.copy()
            mode_val = mode_idx * 8
            infl_val = np.mean([mode_val, high_val])

            # Darkens highlights on very dark backgrounds so they read as part
            # of a hole blob instead of a separate object
            if mode_val < 128:
                _, m = cv2.threshold(layer, infl_val, 255, cv2.THRESH_BINARY)
                m_inv = cv2.bitwise_not(m)
                rev = np.interp(layer, (infl_val, 255), (infl_val, 0))
                rev = rev.astype(np.uint8)
                rev = cv2.bitwise_and(rev, rev, mask=m)
                layer = cv2.bitwise_and(layer, layer, mask=m_inv)
                layer = cv2.add(layer, rev)

            # Rebalances values and adds layer to a combined image
            layer = np.clip(layer, low_val, high_val)
            layer = np.interp(layer, (low_val, high_val), (0, 255))
            layer = layer.astype(np.uint8)
            layer = cv2.bitwise_and(layer, layer, mask=mask)
            layers = cv2.add(layers, layer)

        # Applies global filters to reconstituted image layers
        clahe = cv2.createCLAHE(clipLimit=clahe_limit,
                                tileGridSize=(clahe_grid, clahe_grid))
        layers = clahe.apply(layers)
        layers = cv2.bilateralFilter(layers,
                                     filter_d,
                                     filter_sigma,
                                     filter_sigma)
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        layers = cv2.morphologyEx(layers, cv2.MORPH_OPEN, kernel)
        layers = cv2.morphologyEx(layers, cv2.MORPH_CLOSE, kernel)

        # Obscures logos at the bottom of the target
        logos = (logos * self.out_width / ref_size).astype(int)
        mask = np.zeros((self.out_width, self.out_width), dtype=np.uint8)
        for logo in logos:
            mask = cv2.rectangle(mask, tuple(logo[0]), tuple(logo[1]), 1, -1)
        layers[mask > 0] = np.max(layers)

        # Saves image to class variables and returns status
        self.image_gray = layers
        self.stage_images.append(cv2.cvtColor(layers, cv2.COLOR_GRAY2RGB))
        return None

    def find_shots(self):
        '''
        Uses blob detection to collect keypoint data (position and radius) for
        each shot

        Args:
            None

        Returns:
            None if successful, True if not
        '''
        # Sets blob detection parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = 0
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 600
        params.filterByCircularity = True
        params.minCircularity = .01
        params.filterByConvexity = True
        params.minConvexity = .01
        params.filterByInertia = True
        params.minInertiaRatio = .1

        # Runs blob detection and adds keypoints to image
        detector = cv2.SimpleBlobDetector_create(params)
        self.keypoints = detector.detect(self.image_gray)
        if not self.keypoints:
            return True
        img = cv2.drawKeypoints(self.image,
                                self.keypoints,
                                np.array([]),
                                (0,255,0),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Saves image to class variables and returns status
        self.image = img
        self.stage_images.append(self.image.copy())
        return None

    def get_shot_data(self):
        '''
        Derives shot coordinates and scores from keypoint data (position and
        radius) and gathers all target data into the class DataFrame 'df'

        Args:
            None

        Returns:
            None if successful, True if not
        '''
        # Function settings
        pct_smallest = .2
        overlap_penalty = .85
        max_overlapped = 3

        # Transfers all blob detection keypoint data to a dataframe
        arrow_x = []
        arrow_y = []
        arrow_radii = []
        for k in self.keypoints:
            arrow_x.append(k.pt[0])
            arrow_y.append(k.pt[1])
            arrow_radii.append(k.size / 2)
        df = pd.DataFrame({'x':arrow_x, 'y':arrow_y, 'radius':arrow_radii})

        # Calculates how many shots created a hole based on the mean radius of
        # the smallest holes
        num_smallest = np.ceil(len(arrow_radii) * pct_smallest).astype(int)
        single_size = (np.mean(np.sort(arrow_radii)[:num_smallest]) *
                       overlap_penalty)
        df['count'] = np.clip(df['radius'] // single_size, 0, max_overlapped)
        df['count'] = df['count'].replace(0, 1).astype(int)
        df['id'] = 0
        center = self.out_width // 2

        # Simulates positions of overlapping shots
        if df['count'].max() > 1:
            # Splits dataframe based on which rows represent multiple shots
            clus_df = df[df['count'] > 1].copy()
            df = df[~(df['count'] > 1)].copy()
            clus_df = clus_df.loc[clus_df.index.repeat(clus_df['count'])]
            clus_df['id'] =\
                clus_df.groupby(['x', 'y', 'radius', 'count']).cumcount()

            # Derives a rotation offset for each shot in a cluster
            clus_df['radius'] /= 2
            clus_df['vec_x'] = center - clus_df['x']
            clus_df['vec_y'] = center - clus_df['y']
            clus_df['mag'] = np.sqrt(np.square(clus_df['vec_x']) +
                                        np.square(clus_df['vec_y']))
            clus_df['vec_x'] = (clus_df['vec_x'] / clus_df['mag'] *
                                clus_df['radius'])
            clus_df['vec_y'] = (clus_df['vec_y'] / clus_df['mag'] *
                                clus_df['radius'])
            clus_df['rot'] = np.radians(clus_df['id'] / clus_df['count'] * 360)

            # Calculates new shot coordinates from rotation offset
            clus_df['x'] = (np.cos(clus_df['rot']) * clus_df['vec_x'] -
                            np.sin(clus_df['rot']) * clus_df['vec_y'] +
                            clus_df['x'])
            clus_df['y'] = (np.sin(clus_df['rot']) * clus_df['vec_x'] +
                            np.cos(clus_df['rot']) * clus_df['vec_y'] +
                            clus_df['y'])
            df = df.append(clus_df[['x', 'y', 'radius']], sort=False)

        # Calculates score for each shot
        df['error'] = np.sqrt(np.square(center - df['x']) +
                              np.square(center - df['y']))
        df['score'] = 10 - ((df['error'] - df['radius']) // self.score_step)
        df['score'] = df['score'].clip(0, 10).astype(int)

        # Calculates optimized score for each shot
        grp_center = np.mean(df[['x', 'y']], axis=0)
        df['op_x'] = df['x'] - grp_center['x'] + center
        df['op_y'] = df['y'] - grp_center['y'] + center
        op_error = np.sqrt(np.square(center - df['op_x']) +
                           np.square(center - df['op_y']))
        df['op_score'] = 10 - ((op_error - df['radius']) // self.score_step)
        df['op_score'] = df['op_score'].clip(0, 10).astype(int)

        # Sorts dataframe and drops columns that are no longer necessary
        df.sort_values(['error'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(['count', 'id'], axis=1, inplace=True)

        # Saves dataframe to class variable and returns status
        self.df = df
        return None
