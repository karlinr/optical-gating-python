from loguru import logger
from app.config import Config

class TriggerDecider:
    def __init__(self):
        self.frame_interval = 1.0 / Config.Cameras.BF.framerate
        self.most_recent_trigger_time = -10000
    
    def evaluate_trigger(self, current_time, predicted_time, est_period):
        """
        Translates absolute predicted targets into relative lookahead values 
        to evaluate the original spim-interface triggering criteria.
        
        Returns:
            (bool, float): A tuple containing a flag indicating whether to fire, 
                           and the final adjusted relative time to wait.
        """
        time_to_wait_s = predicted_time - current_time

        if self.most_recent_trigger_time >= current_time - (est_period / 2.0):
            logger.debug("Trigger rejected: Already issued on this cardiac cycle.")
            # Coarsely shift target prediction to the next expected heartbeat cycle
            time_to_wait_s += est_period
            return False, time_to_wait_s

        if time_to_wait_s < Config.Gating.PREDICTION_LATENCY:
            # Panic mode: Target is extremely close, but fire immediately anyway
            logger.warning(f"Panic trigger issued! Lookahead ({time_to_wait_s:.4f}s) is below latency floor.")
            self.most_recent_trigger_time = current_time + time_to_wait_s
            return True, time_to_wait_s
            
        if (time_to_wait_s - (Config.Gating.EXTRAPOLATION_FACTOR * self.frame_interval)) < Config.Gating.PREDICTION_LATENCY:
            # Standard commit window: Not enough time to wait for the next frame's data
            logger.debug(f"Standard trigger committed. Lookahead: {time_to_wait_s:.4f}s.")
            self.most_recent_trigger_time = current_time + time_to_wait_s
            return True, time_to_wait_s

        logger.debug(f"Hold trigger: Lookahead ({time_to_wait_s:.4f}s) allows waiting for the next frame.")
        return False, time_to_wait_s

    def handle_hardware_rejection(self, current_time, est_period):
        """
        Forces a software cooldown following a hardware timing box rejection.
        Prevents rapid-fire panic cycles within a heartbeat that's already passed.
        """
        logger.warning(
            f"Hardware collision detected! Forcing a software lockout cooldown "
            f"for the remaining cycle duration ({est_period:.4f}s)."
        )
        # Advance the trigger lockout to the future to block subsequent frames in this cycle
        self.most_recent_trigger_time = current_time