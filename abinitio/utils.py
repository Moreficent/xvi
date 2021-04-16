def fix_seed(seed):
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import tensorflow
    tensorflow.random.set_seed(seed)
