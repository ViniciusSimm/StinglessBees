# GRAPHIC CARD

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# import tensorflow as tf
print(tf.__version__)

# import numpy as np
# import matplotlib.pyplot as plt
# import sklearn.metrics

# valores = [[27,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#            [ 0,102,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  0],
#            [ 0,  0, 75,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0],
#            [ 0,  3,  0,260,  0,  1,  0,  0,  1,  0,  0,  0,  1],
#            [ 0,  1,  0,  2, 80,  0,  0,  0,  0,  0,  0,  1,  0],
#            [ 0,  3,  0,  3,  0, 41,  1,  0,  0,  0,  0,  0,  1],
#            [ 0,  1,  3,  0,  0,  1,194,  2,  0,  0,  1,  0,  0],
#            [ 0,  0,  1,  3,  0,  0,  1, 44,  0,  0,  0,  0,  0],
#            [ 0,  0,  1,  0,  0,  0,  1,  0, 54,  0,  0,  0,  0],
#            [ 0,  0,  2,  0,  0,  0,  0,  0,  0, 39,  0,  1,  0],
#            [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0, 41,  0,  0],
#            [ 0,  2,  0,  0,  0,  0,  1,  0,  0,  0,  0, 45,  1],
#            [ 0,  0,  1,  1,  0,  2,  0,  0,  1,  1,  0,  0, 50]]

# sum_cm = np.array(valores)

# disp = sklearn.metrics.ConfusionMatrixDisplay(sum_cm,display_labels=['boca_de_sapo', 'tubuna', 'bora', 'jatai', 'mandaguari', 'mirim_droryana', 'mandacaia', 'bugia', 'lambe_olhos', 'japura', 'moca_branca', 'irai', 'mirim_preguica'])

# fig, ax = plt.subplots(figsize=(12, 10))

# disp.plot(ax=ax,cmap=plt.cm.Blues)
# plt.xticks(rotation=45)
# plt.show()