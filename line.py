import numpy as np

class Line():

    def __init__(self):
        self.n = 30

        # was the line detected in the last iteration?
        self.detected = False  
        #xbase values of the last n fits of the line
        self.recent_basis = [] 
        #average xbase values of fitted line over last n iterations
        self.best_xbase = None
        #last n sets of line coefficients
        self.last_n_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])] 
        self.current_metric_fit = [np.array([False])]
        #radius of curvature of the line in pixel space
        self.roc_pix = None 
        #radius of curvature of the line in metric space
        self.roc_metric = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def set_current_fit(self, fit):
        self.last_n_fits.append(fit)
        if len(self.last_n_fits) > self.n:
            self.last_n_fits = self.last_n_fits[1:]

        self.current_fit = fit

        self.best_fit = np.mean(self.last_n_fits, axis=0)

    def set_current_metric_fit(self, fit):
        self.current_metric_fit = fit
 
    def get_current_xbase(self):
        return self.recent_basis[-1]

    def set_base_value(self, xbase):
        self.recent_basis.append(xbase)

        if len(self.recent_basis) > self.n:
            self.recent_basis  = self.recent_basis[1:]

        self.best_xbase  = \
            int(np.mean(self.recent_basis , axis=0))


        

