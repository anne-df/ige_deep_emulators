import jax
import jax.numpy as jnp
from flax import linen as nn

class UNet(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        def conv_block(x, out_channels):
            x = nn.Conv(features=out_channels, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.Conv(features=out_channels, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
            return x

        enc1 = conv_block(x, 64)
        enc2 = conv_block(nn.max_pool(enc1, (2, 2), strides=(2, 2)), 128)
        enc3 = conv_block(nn.max_pool(enc2, (2, 2), strides=(2, 2)), 256)
        enc4 = conv_block(nn.max_pool(enc3, (2, 2), strides=(2, 2)), 512)

        bottleneck = conv_block(nn.max_pool(enc4, (2, 2), strides=(2, 2)), 1024)

        dec4 = nn.ConvTranspose(features=512, kernel_size=(2, 2), strides=(2, 2))(bottleneck)
        dec4 = jnp.concatenate([dec4, enc4], axis=-1)
        dec4 = conv_block(dec4, 512)

        dec3 = nn.ConvTranspose(features=256, kernel_size=(2, 2), strides=(2, 2))(dec4)
        dec3 = jnp.concatenate([dec3, enc3], axis=-1)
        dec3 = conv_block(dec3, 256)

        dec2 = nn.ConvTranspose(features=128, kernel_size=(2, 2), strides=(2, 2))(dec3)
        dec2 = jnp.concatenate([dec2, enc2], axis=-1)
        dec2 = conv_block(dec2, 128)

        dec1 = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2))(dec2)
        dec1 = jnp.concatenate([dec1, enc1], axis=-1)
        dec1 = conv_block(dec1, 64)

        return nn.Conv(features=self.out_channels, kernel_size=(1, 1))(dec1)

# Example usage:
# model = UNet(in_channels=3, out_channels=1)
# variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128, 128, 3)))
# output = model.apply(variables, jnp.ones((1, 128, 128, 3)))
# print(output.shape)


class UNet1D(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        def conv_block(x, out_channels):
            x = nn.Conv(features=out_channels, kernel_size=(3,), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.Conv(features=out_channels, kernel_size=(3,), padding='SAME')(x)
            x = nn.relu(x)
            return x

        enc1 = conv_block(x, 64)
        enc2 = conv_block(nn.max_pool(enc1, (2,), strides=(2,)), 128)
        enc3 = conv_block(nn.max_pool(enc2, (2,), strides=(2,)), 256)
        enc4 = conv_block(nn.max_pool(enc3, (2,), strides=(2,)), 512)

        bottleneck = conv_block(nn.max_pool(enc4, (2,), strides=(2,)), 1024)

        dec4 = nn.ConvTranspose(features=512, kernel_size=(2,), strides=(2,))(bottleneck)
        dec4 = jnp.concatenate([dec4, enc4], axis=-1)
        dec4 = conv_block(dec4, 512)

        dec3 = nn.ConvTranspose(features=256, kernel_size=(2,), strides=(2,))(dec4)
        dec3 = jnp.concatenate([dec3, enc3], axis=-1)
        dec3 = conv_block(dec3, 256)

        dec2 = nn.ConvTranspose(features=128, kernel_size=(2,), strides=(2,))(dec3)
        dec2 = jnp.concatenate([dec2, enc2], axis=-1)
        dec2 = conv_block(dec2, 128)

        dec1 = nn.ConvTranspose(features=64, kernel_size=(2,), strides=(2,))(dec2)
        dec1 = jnp.concatenate([dec1, enc1], axis=-1)
        dec1 = conv_block(dec1, 64)

        return nn.Conv(features=self.out_channels, kernel_size=(1,))(dec1)

# Example usage:
# n_data = 10
# in_channels = 5
# vector_size = 128
# model = UNet1D(in_channels=in_channels, out_channels=1)
# app_data = jnp.ones((n_data, vector_size, in_channels))
# variables = model.init(jax.random.PRNGKey(0), app_data)
# output = model.apply(variables, app_data)
# print(output.shape)




# Basic UNet model

class UNet1D_base(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        def conv_block(x, out_channels):
            x = nn.Conv(features=out_channels, kernel_size=(3,), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.Conv(features=out_channels, kernel_size=(3,), padding='SAME')(x)
            x = nn.relu(x)
            return x

        enc1 = conv_block(x, 64)

        bottleneck = conv_block(nn.max_pool(enc1, (2,), strides=(2,)), 128)

        dec1 = nn.ConvTranspose(features=64, kernel_size=(2,), strides=(2,))(bottleneck)
        dec1 = jnp.concatenate([dec1, enc1], axis=-1)
        dec1 = conv_block(dec1, 64)

        return nn.Conv(features=self.out_channels, kernel_size=(1,))(dec1)
    


# Basic UNet model 2x2 layers

class UNet1D_double(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        def conv_block(x, out_channels):
            x = nn.Conv(features=out_channels, kernel_size=(3,), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.Conv(features=out_channels, kernel_size=(3,), padding='SAME')(x)
            x = nn.relu(x)
            return x

        enc1 = conv_block(x, 64)
        enc2 = conv_block(nn.max_pool(enc1, (2,), strides=(2,)), 128)

        bottleneck = conv_block(nn.max_pool(enc2, (2,), strides=(2,)), 256)

        dec2 = nn.ConvTranspose(features=128, kernel_size=(2,), strides=(2,))(bottleneck)
        dec2 = jnp.concatenate([dec2, enc2], axis=-1)
        dec2 = conv_block(dec2, 128)

        dec1 = nn.ConvTranspose(features=64, kernel_size=(2,), strides=(2,))(dec2)
        dec1 = jnp.concatenate([dec1, enc1], axis=-1)
        dec1 = conv_block(dec1, 64)

        return nn.Conv(features=self.out_channels, kernel_size=(1,))(dec1)