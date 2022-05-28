from segmentation_models.base import functional as F

def dice_coef(y_true, y_pred):
    """
    Dice coefficient

    # Source https://www.kaggle.com/code/awsaf49/uwmgi-transunet-2-5d-train-tf
    """
    dice = F.f_score(
        y_true,
        y_pred,
        beta=1,
        smooth=1e-5,
        per_image=False,
        threshold=0.5,
        **kwargs,
    )
    return dice


def tversky(y_true, y_pred, axis=(0, 1, 2), alpha=0.3, beta=0.7, smooth=0.0001):
    """
    Tversky metric
    
    # Source https://www.kaggle.com/code/awsaf49/uwmgi-transunet-2-5d-train-tf
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.math.reduce_sum(y_true * y_pred, axis=axis) # calculate True Positive
    fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=axis) # calculate False Negative
    fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=axis) # calculate False Positive
    tv = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth) # calculate tversky
    tv = tf.math.reduce_mean(tv)
    return tv


def tversky_loss(y_true, y_pred):
    """
    Tversky Loss
    
    # Source https://www.kaggle.com/code/awsaf49/uwmgi-transunet-2-5d-train-tf
    """
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    """
    Focal Tversky Loss: Focal Loss + Tversky Loss
    
    # Source https://www.kaggle.com/code/awsaf49/uwmgi-transunet-2-5d-train-tf
    """
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)
