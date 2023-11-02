import os


for activation in ["phi"]:

    for kernel_initializer in ["Glorot-normal"]:
            
        for model in ["mscalenn2","mscalenn2-multi-alphai","mscalenn2-multi-alphai-ci"]:
            
            os.system('python -m src.train --n_epoch=10  --model={} --activation={} --kernel_initializer={}'.format(model,activation,kernel_initializer) )


        os.system('python -m src.train_fnn --n_epoch=10  --activation={} --kernel_initializer={}'.format(activation,kernel_initializer) )


