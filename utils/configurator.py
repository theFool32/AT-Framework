class Configurator:
    def __new__(self):
        if not hasattr(self, "instance"):
            self.instance = super().__new__(self)
        return self.instance

    def __setattr__(self, key, value):
        self.instance.__dict__[key.lower()] = value
        return self.instance

    def __getitem__(self, key):
        return self.instance.__dict__[key.lower()]

    def update(self, params):
        for k, v in params.items():
             self.instance.__setattr__(k, v)
        return self.instance

    def __repr__(self):
        return repr(self.instance.__dict__)

if __name__ == '__main__':
    a = {'a': 1, 'b':2, 'c':3}
    Configurator().update(a)
    assert Configurator().a == 1
