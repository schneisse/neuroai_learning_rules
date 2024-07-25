import torch

# torch.autograd.set_detect_anomaly(True)


def MSE(y, y_pred):
    return 0.5 * (y - y_pred) ** 2


class PredictiveLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, mu: torch.Tensor):
        if self.training:
            if self.x is None:
                x_data = mu.detach().clone()
                x = torch.nn.Parameter(x_data.to(mu.device), True)
                self.x = x

            energy = MSE(mu, self.x)
            self.energy = energy.sum()
            return self.x
        else:
            return mu


class PredictiveTrainer:
    def __init__(
        self,
        T: int,
        model: torch.nn.Module,
        optimizer_x_fn,
        optimizer_p_fn,
        x_lr,
        p_lr,
    ):
        self.T = T
        self.model = model
        self.optimizer_x_fn = optimizer_x_fn
        self.optimizer_p_fn = optimizer_p_fn
        self.x_lr = x_lr
        self.p_lr = p_lr

        self.optimizer_p = None
        self.optimizer_x = None

    def get_pc_layers(self):
        for module in self.model.modules():
            if isinstance(module, PredictiveLayer):
                yield module

    def get_internal_representations(self):
        for pc_layer in self.get_pc_layers():
            model_x = pc_layer.x
            if model_x is not None:
                yield model_x

    def get_weights(self):
        internal_reps = self.get_internal_representations()
        for param in self.model.parameters():
            if not any(param is x for x in internal_reps):
                yield param

    def create_optimizer_x(self):
        self.optimizer_x: torch.optim.Optimizer = self.optimizer_x_fn(
            self.get_internal_representations(), lr=self.x_lr
        )

    def create_optimizer_p(self):
        self.optimizer_p: torch.optim.Optimizer = self.optimizer_p_fn(
            self.get_weights(), lr=self.p_lr
        )

    def train_batch(self, inputs, label, loss_fn):
        results = {
            "loss": [],
            "energy": [],
            "combined": [],
        }

        for t in range(self.T):

            if t == 0:
                for pc_layer in self.get_pc_layers():
                    pc_layer.x = None
                    # setting it to None initialises x with mu
            outputs = self.model(inputs).clone()

            if t == 0:
                self.create_optimizer_x()
                if self.optimizer_p is None:
                    self.create_optimizer_p()

            loss = loss_fn(outputs, label)
            energy = sum([pc_layer.energy for pc_layer in self.get_pc_layers()])

            combined = sum([loss, energy])

            if t == self.T - 1:
                self.optimizer_p.zero_grad()

            self.optimizer_x.zero_grad()

            combined.backward()

            self.optimizer_x.step()
            if t == self.T - 1:
                self.optimizer_p.step()

            results["loss"].append(loss.item())
            results["energy"].append(energy.item())
            results["combined"].append(combined.item())

        return results
