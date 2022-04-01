import torch

class Accumulator:
    def __init__(self, dyn_lambda=None):
        self.dyn_lambda = dyn_lambda
        self.cumulated = None
        self.counter = 0

        self.reset()

    def reset(self):
        if self.dyn_lambda is None:
            # Averaging..
            self.cumulated = None
            self.counter = 0

    def add_list(self, d_list):
        assert isinstance(d_list, list)

        all_d = torch.cat(d_list, dim=0)
        if all_d.size(0) == 0:
            d = torch.zeros(*all_d.size()[1:]).to(device=all_d.get_device())
        else:
            d = all_d.mean(dim=0)

        self.add(d)

    def add(self, d):
        if self.cumulated is None:
            self.cumulated = d
        else:
            if self.dyn_lambda is None:
                self.cumulated += d
            else:
                self.cumulated = self.dyn_lambda * self.cumulated + (1 - self.dyn_lambda) * d

        self.counter += 1

    def get(self):
        if self.dyn_lambda is None:
            assert self.counter > 0
            return self.cumulated / self.counter
        else:
            return self.cumulated.clone()
