from typing import NamedTuple, Dict

class CrossValidationParameters(NamedTuple):
    """Parameters to set for scikit learn splitters"""
    n_splits: int = 5
    random_state: int = 0
    shuffle: bool = True

class TrainingParameters(NamedTuple):
    """General parameters to set for model training"""
    n_epochs: int = 100
    learning_rate: float = 0.0003
    batch_size: int = 256
    patience: int = 20
    seed: int = 0

class ModelParameters(NamedTuple):
    """Parameters for the protgoat model"""
    dropout_rate: float = 0.5
    l1_dim: int = 600
    l2_dim: int = 300
    alpha: float = 0.1 # negative slope of leaky relu
    final_dim: int = 800

    @property
    def branch_kwargs(self) -> Dict:
        return dict(l1_dim=self.l1_dim, 
                    l2_dim=self.l2_dim, 
                    dropout_rate=self.dropout_rate, 
                    alpha=self.alpha)
    
class ContrastiveModelParameters(NamedTuple):
    """Parameters for the contrastive learning model"""
    dropout_rate_1: float = 0.5
    hidden_dim: int = 600
    dropout_rate_2: float = 0.5

    l2_dim: int = 300
    alpha: float = 0.1 # negative slope of leaky relu
    final_dim: int = 800

    @property
    def branch_kwargs(self) -> Dict:
        return dict(l1_dim=self.l1_dim, 
                    l2_dim=self.l2_dim, 
                    dropout_rate=self.dropout_rate, 
                    alpha=self.alpha)