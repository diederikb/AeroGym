from gymnasium import Wrapper

class ResetOptions(Wrapper):
    def __init__(self, env, reset_options):
        super().__init__(env)
        self.reset_options = reset_options

    def reset(self, **kwargs):
        obs, info = self.env.reset(options=self.reset_options, **kwargs)
        return obs, info
