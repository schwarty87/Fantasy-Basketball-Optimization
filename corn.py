from typing import List
import torch
import torch.nn.functional as F


def corn_loss(y_true, logits, num_classes, class_weights=None):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.

    y_true : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    class_weights : np.ndarray
        weights for each of the classes

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.

    Examples
    ----------
    import torch
    from coral_pytorch.losses import corn_loss
    # Consider 8 training examples
    _  = torch.manual_seed(123)
    X_train = torch.rand(8, 99)
    y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    NUM_CLASSES = 5
    #
    #
    # def __init__(self):
    corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    #
    #
    # def forward(self, X_train):
    logits = corn_net(X_train)
    logits.shape
    torch.Size([8, 4])
    corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    class_vals = torch.arange(end=num_classes-1, dtype=y_true.dtype)
    set_masks = y_true[..., None] > (class_vals - 1)
    label_tensors = y_true[..., None] > class_vals
    logits_sigmoid = F.logsigmoid(logits)
    losses = torch.where(set_masks,
                         torch.where(label_tensors,
                                     logits_sigmoid,           # if label > i
                                     logits_sigmoid - logits,  # if label <= i
                                     ),
                         0.0  # don't add to loss if label is <= i - 1
                         )
    losses *= -1

    if class_weights is not None:
        losses *= class_weights

    return torch.sum(losses, dim=-1) / torch.sum(set_masks, dim=-1)  # num_classes #


class CornLoss(torch.nn.Module):
    """
    Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Examples
    ----------
     import torch
     from coral_pytorch.losses import corn_loss
     # Consider 8 training examples
     _  = torch.manual_seed(123)
     X_train = torch.rand(8, 99)
     y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
     NUM_CLASSES = 5
     #
     #
     # def __init__(self):
     corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
     #
     #
     # def forward(self, X_train):
     logits = corn_net(X_train)
     logits.shape
    torch.Size([8, 4])
     corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, y_true, logits, sample_weights):
        """
        Parameters
        ----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
            Outputs of the CORN layer.

        y_true : torch.tensor, shape=(num_examples)
            Torch tensor containing the class labels.

        Returns
        ----------
        loss : torch.tensor
            A torch.tensor containing a single loss value.
        """
        y_true = torch.squeeze(y_true)
        return corn_loss(y_true, logits, num_classes=self.num_classes)


@torch.jit.script
def corn_logits_to_cumprobs(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cum_probs = torch.sigmoid(logits)
    cum_probs = torch.cumprod(cum_probs, dim=dim)
    return cum_probs


def corn_logits_to_pdf(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cum_probs = corn_logits_to_cumprobs(logits=logits, dim=dim)
    return cumprobs_to_pdf(cum_probs)


def corn_logits_to_label(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Returns the predicted rank label from logits for a
    network trained via the CORN loss.

    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes)
        Torch tensor consisting of logits returned by the
        neural net.
    dim : int, the dimension to work on

    Returns
    ----------
    labels : torch.tensor, shape=(n_examples)
        Integer tensor containing the predicted rank (class) labels


    Examples
    ----------
    # 2 training examples, 5 classes
    logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
    ...                        [65.667, 0.303, 11.500, -4.524]])
    corn_label_from_logits(logits)
    tensor([1, 3])
    """

    pdf = corn_logits_to_pdf(logits, dim=dim)
    predicted_labels = torch.sum(torch.arange(pdf.shape[-1]) * pdf, dim=-1)
    # pdf2 = pdf ** 2
    # predicted_labels = torch.div(torch.sum(torch.arange(pdf.shape[-1]) * pdf2, dim=-1), torch.sum(pdf2, dim=-1))
    # # sum of the cumsums that are greater than 0.5
    # predict_levels = cum_probs > 0.5
    # predicted_labels = torch.sum(predict_levels, dim=dim) + 1
    return predicted_labels


def corn_logits_to_labels_list(y_pred: List[torch.Tensor]):
    return torch.stack([corn_logits_to_label(yp) for yp in y_pred], dim=2)


def cumprobs_to_pdf(cumprobs: torch.Tensor) -> torch.Tensor:
    """Turns ordinal probabilities into label probabilities (softmax)."""

    # Number of columns is the number of classes - 1
    num_classes = cumprobs.shape[-1] + 1

    # Create a list of tensors.
    probs = []

    # First, get probability predictions out of the cumulative logits.
    # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
    # Pr(Y = 0) = 1 - s(logit for column 0)
    probs.append((1.0 - cumprobs[..., 0])[..., None])

    # For the other columns, the probability is:
    # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
    if num_classes > 2:
        probs.append(cumprobs[..., :-1] - cumprobs[..., 1:])
        # for val in range(1, num_classes - 1):
        #     probs.append(cumprobs[..., val - 1] - cumprobs[..., val])

    # Special handling of the maximum label value.
    probs.append(cumprobs[..., num_classes - 2][..., None])

    # Combine as columns into a new tensor.
    probs_tensor = torch.concat(probs, dim=-1)

    return probs_tensor
