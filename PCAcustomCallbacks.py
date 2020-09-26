# %%
class PCAGeneralizer(keras.callbacks.Callback):
    def __init__(self,var2explain):
        """ Save params in constructor
        """
        self.var2explain = var2explain

    def on_train_batch_end(self , batch, logs=None):
        var2explain = self.var2explain
        weights = model.layers[1].get_weights()[0]
        weightsB = model.layers[1].get_weights()[1]           
        mn = tf.reduce_mean(weights,0)
        weights_normalized = weights - mn
        S, U, Vt = svd(weights_normalized, full_matrices=True)
        eigvalSVD = []
        n=weights.shape[0]
        eigvalSVD = (S ** 2) / (n - 1)
        varExpRatio_tf = eigvalSVD / sum(eigvalSVD)
        varExpCum_tf = tf.cumsum(varExpRatio_tf)
        num_comps_tf = len(varExpCum_tf[varExpCum_tf<var2explain]) + 1
        comps_tf = ((Vt)[:,:num_comps_tf])

        X_reduced_tf = tf.matmul(weights_normalized, comps_tf)
        weights_reproduced = tf.tensordot(X_reduced_tf,tf.transpose(comps_tf),axes=1) + mn
        model.layers[1].set_weights([weights_reproduced,weightsB])
        # print("weights shape            --->", weights.shape)
        # print(weights[0,0])
        # print("weights_reproduced shape --->", weights_reproduced.shape)
        
        f.write(str(num_comps_tf))
        f.write(",")
        f.write(str(logs.get('loss')))
        f.write(",")
        f.write(str(logs.get('accuracy')))
        f.write("\n")