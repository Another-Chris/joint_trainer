## 2022-11-7
1 alpha = 0.5, discriminator(units = 512), min(EER) = 10.2
2 alpha = 0.5, discriminator(units = 1024), min(EER) = 10.8
3 alpha = 0.6, discriminator(units = 512), lr = 1e-4, batch_size = 200, steps = 256, min(EER) = 9.99 **change things after this** 
4 alpha = 0.6, discriminator(units = 1024), min(EER) = bull shit
5 alpha = 0.7, discriminator(units = 1024), min(EER) = bull shit
6 alpha = 0.8, discriminator(units = 1024), min(EER) = bull shit, the EER actually increases
so try to decrease the power of discriminator.
but before that, try with only one segment.
alpha = 0.6, discriminator(units = 512), with only one segment. min(EER) = ~10.9
alpha = 0.5, discriminator(units = 512), with only one segment, min(EER) = 
alpha = 0.6, discriminator(units = 1024),with only one segment, min(EER) = 11.01

## 2022-11-11
what do you need to fine-tune?
discriminator: discriminator + alpha
data: normalize or not, two segments or not
based on the best you have. 
so it's 3, but change the normalization.

one segments you haven't had the best results yet
but two segments you do.


changing the learning rate and smaller the step size
alpha = 0.6, discriminator(units = 512), lr = 5e-5, batch_size = 200, steps = 128, min(EER) = 9.687
alpha = 0.6, discriminator(units = 512), lr = 1e-5, batch_size = 200, steps = 128, min(EER) = 