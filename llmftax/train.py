from dataclasses import dataclass
from typing import Callable, Any
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

@dataclass
class EncoderTrainer:
    model: Any  # Replace with the specific model type
    optimizer: optax.GradientTransformation
    scheduler: Callable[[int], float]
    metric: Callable[[jnp.ndarray, jnp.ndarray], float]
    criterion: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
    verbose: bool = True

    def train(self, state: train_state.TrainState, data_loader):
        """Train the model for one epoch."""
        total_loss = 0
        all_predictions = []
        all_labels = []

        for batch in data_loader:
            state, loss, predictions, labels = self.process_batch(state, batch, train=True)
            total_loss += loss
            all_predictions.append(predictions)
            all_labels.append(labels)

        return self.finalize_epoch(total_loss, all_predictions, all_labels, len(data_loader))

    def evaluate(self, state: train_state.TrainState, data_loader):
        """Evaluate the model on a validation set."""
        total_loss = 0

        for batch in data_loader:
            _, loss, _, _ = self.process_batch(state, batch, train=False)
            total_loss += loss

        average_loss = total_loss / len(data_loader)
        return average_loss

    def process_batch(self, state: train_state.TrainState, batch, train=True):
        """Process a single batch of data."""
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        def loss_fn(params):
            logits = self.model.apply({'params': params}, input_ids, attention_mask)
            criterion_mode = getattr(self.criterion, 'mode', None)
            if criterion_mode == 'input' and 'type_indicator' in batch:
                type_indicator = batch['type_indicator']
                loss = self.criterion(logits, labels, type_indicator)
            else:
                loss = self.criterion(logits, labels)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        
        if train:
            updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            state = state.apply_gradients(grads=grads, opt_state=opt_state)
            lr = self.scheduler(state.step)
            state = state.replace(step=state.step + 1, lr=lr)

        predictions = jnp.argmax(logits, axis=1) if train else None
        return state, loss, predictions, labels

    def finalize_epoch(self, total_loss, all_predictions, all_labels, num_batches):
        """Finalize epoch, calculate metrics and average loss."""
        all_predictions = jnp.concatenate(all_predictions)
        all_labels = jnp.concatenate(all_labels)
        metric_output = self.metric(all_predictions, all_labels)
        average_loss = total_loss / num_batches
        if self.verbose:
            print(f'Epoch finished. Average Loss: {average_loss:.4f}, Metric: {metric_output}')
        return average_loss, metric_output
