Works

But has seemingly bluescreened my computer with a directx crash so

Perhaps not the most stable working?

But it sure works good.

大大成功

Notes
-----

Resnet:
    Does not train nearly as fast as Unet. Even at epoch 100 had not alltogether satisfactory results.
    Used epoch 55, still had a worse result compared to resnet at epoch 21

Densenet: 
    works, possibly better than resnet

Densenet with conv skip layers:
    Doesn't work

Deformable Densenet:
    Works, slightly better than densenet
    But also is deathly slow :skull: like 6x slower

All of these, discriminator accuracy at 100 ~ generator too weak

Densenet:
    Changing the lambda reconstruction to 5 (putting adversarial doubling) - run 12 doesnt do much
    Changing adversarial to use log loss - run 13 - doesn't converge
    Changing to mean squared log error - run 14 - working... kinda works, nothing special.

Plans:

- Try Deformable Conv Layer in UNet DONE
- Try ResNet DONE
