from agent.controllers.DreamerController import DreamerController
from configs.dreamer.mamujoco.mamujocoAgentConfig import MAMujocoDreamerConfig

class MAMujocoDreamerControllerConfig(MAMujocoDreamerConfig):
    def __init__(self):
        super().__init__()

        self.epsilon = 0. # 0.05
        self.EXPL_DECAY = 0.9999
        self.EXPL_NOISE = 0.
        self.EXPL_MIN = 0.
        
        self.temperature = 1.
        self.determinisitc = False

    def create_controller(self):
        return DreamerController(self)
