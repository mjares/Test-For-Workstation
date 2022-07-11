class OptimalBlendedController:
    def __init__(self):
        self.current_mode = ''
        self.limits = {'Rotor': 0.1, 'AttNoise': 0.4}

    def step(self, mode, fault_mag):
        if mode == 'Rotor' and fault_mag > self.limits[mode]:
            weight = 0
        elif mode == 'AttNoise' and fault_mag > self.limits[mode]:
            weight = 1
        else:
            weight = 0.5
        return weight
