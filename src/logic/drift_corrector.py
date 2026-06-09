import numpy as np
from loguru import logger
from logic.utils import sad_with_references
from app.config import Config

class DriftCorrector:
    """ Drift corrector adapted from open-optical-gating"""

    def __init__(self):
        self.drift = (0, 0)

    def add_sample(self, frame, best_match = None, refs = None, matching_frame = None, dxRange=range(-30,31,3), dyRange=range(-30,31,3)):
        if Config.Gating.DRIFT_CORRECT:
            if best_match is not None:
                self.drift = self._update_drift_estimate(frame, best_match, self.drift)
            elif refs is not None:
                self.drift = self._get_drift_estimate(frame, refs, matching_frame=matching_frame, dxRange=dxRange, dyRange=dyRange)
            else:
                logger.warning("DriftCorrector.add_sample called without target frame or references.")
                
            return self.drift
        else:
            return (0, 0)

    def adjust_reference_array(self, array, drift=None):
        if Config.Gating.DRIFT_CORRECT:
            if drift is None:
                drift = self.drift
            dx, dy = drift
            mx, my = abs(dx), abs(dy)
            
            if mx == 0 and my == 0:
                return array
                
            w_idx = array.ndim - 2
            h_idx = array.ndim - 1
            W = array.shape[w_idx]
            H = array.shape[h_idx]
            
            slices = [slice(None)] * array.ndim
            slices[w_idx] = slice(mx, W - mx)
            slices[h_idx] = slice(my, H - my)
            
            return array[tuple(slices)]
        else:
            return array

    def adjust_live_frame(self, frame, drift=None):
        if Config.Gating.DRIFT_CORRECT:
            if drift is None:
                drift = self.drift
            dx, dy = drift
            mx, my = abs(dx), abs(dy)
            
            if mx == 0 and my == 0:
                return frame
                
            W, H = frame.shape
            
            start_x = mx - dx
            end_x = W - mx - dx
            start_y = my - dy
            end_y = H - my - dy
            
            return frame[start_x:end_x, start_y:end_y]
        else:
            return frame
    
    def _get_drift_estimate(self, frame, refs, matching_frame=None, dxRange=range(-30,31,3), dyRange=range(-30,31,3)):
        """ Determine an initial estimate of the sample drift.
            We do this by trying a range of variations on the relative shift between frame0 and the best-matching frame in the reference sequence.
            
            Parameters:
                frame          array-like      2D frame pixel data for the frame we should use
                refs           list of arrays  List of 2D reference frame pixel data that we should search within
                matching_frame int             Entry within reference frames that is the best match to 'frame',
                                            or None if we don't know what the best match is yet
                dxRange        list of int     Candidate x shifts to consider
                dyRange        list of int     Candidate y shifts to consider
            
            Returns:
                new_drift      (int,int)       New drift parameters
            """
        # frame0 and the images in 'refs' must be numpy arrays of the same size
        assert frame.shape == refs[0].shape
        
        # Identify region within bestMatch that we will use for comparison.
        # The logic here basically follows that in phase_matching, but allows for extra slop space
        # since we will be evaluating various different candidate drifts
        inset = np.maximum(np.max(np.abs(dxRange)), np.max(np.abs(dyRange)))
        rect = [
                inset,
                frame.shape[0] - inset,
                inset,
                frame.shape[1] - inset,
                ]  # X1,X2,Y1,Y2
                
        candidateShifts = []
        for _dx in dxRange:
            for _dy in dyRange:
                candidateShifts += [(_dx,_dy)]

        if matching_frame is None:
            ref_index_to_consider = range(0, len(refs))
        else:
            ref_index_to_consider = [matching_frame]

        # Build up a list of frames, each representing a window into frame with slightly different drift offsets
        frames = []
        for shft in candidateShifts:
            dxp = shft[0]
            dyp = shft[1]
            
            # Adjust for drift and shift
            rectF = np.copy(rect)
            rectF[0] -= dxp
            rectF[1] -= dxp
            rectF[2] -= dyp
            rectF[3] -= dyp
            frames.append(frame[rectF[0] : rectF[1], rectF[2] : rectF[3]])

        # Compare all these candidate shifted images against each of the candidate reference frame(s) in turn
        # Our aim is to find the best-matching shift from within the search space
        best = 1e200
        for r in ref_index_to_consider:
            sad = sad_with_references(refs[r][rect[0] : rect[1], rect[2] : rect[3]], np.array(frames))
            smallest = np.min(sad)
            if (smallest < best):
                bestShiftPos = np.argmin(sad)
                best = smallest

        return (candidateShifts[bestShiftPos][0],
                candidateShifts[bestShiftPos][1])

    def _update_drift_estimate(self, frame0, bestMatch0, drift0):
        """ Determine an updated estimate of the sample drift.
            We try changing the drift value by ±1 in x and y.
            This just calls through to the more general function get_drift_estimate()
            
            Parameters:
                frame0         array-like      2D frame pixel data for our most recently-received frame
                bestMatch0     array-like      2D frame pixel data for the best match within our reference sequence
                drift0         (int,int)       Previously-estimated drift parameters
            Returns
                new_drift      (int,int)       New drift parameters
            """
        # Note that these parameters mean we consider the current drift value first, before trying different values.
        # The search order shouldn't make any difference in most situations, but this way the drift values won't go
        # beyond the frame width/height if the drift values diverge.
        # I think the situation where the drift diverges is unrecoverable with the code in its current form,
        # but this way we avoid ever-increasing values for drift. That seems to have led to a memory-related crash in the past,
        # though I haven't managed to figure out exactly what code was crashing.
        return self._get_drift_estimate(frame0, [bestMatch0], dxRange=drift0[0]+np.array([0,-1,1]), dyRange=drift0[1]+np.array([0,-1,1]))
