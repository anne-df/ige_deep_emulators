from flax import linen as nn

# MLP Model Definition with Flax
class MLP(nn.Module):
    hidden_sizes: list

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)  # Reshape to (batch_size, 5 * depth)
        for size in self.hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.tanh(x)
        x = nn.Dense(features=50)(x)  # Output layer
        return x