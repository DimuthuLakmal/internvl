from transformers import Trainer
import numpy as np
import pickle
import torch.nn.functional as F
import torch
from transformers import PreTrainedModel
from typing import Optional, Dict, Union, List


class DistillSFTTrainer(Trainer):

    def __init__(
        self,
        logits_path = None,
        teacher_vocab_size = None,  
        kd_ratio: float = 0.25,    
        max_seq_length : int = 1024,
        distillation_type: str = "forward_kld",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher_vocab_size = teacher_vocab_size
        self.kd_ratio = kd_ratio
        self.max_seq_length = max_seq_length
        self.distillation_type = distillation_type

        with open(logits_path, "rb") as f:
            self.teacher_logits = pickle.load(f)


    def _load_teacher_logits(self, batch_size: int, it: int, dp_rank: int, device: torch.device, no_model_batch: Dict, idx: None):
        start_idx = dp_rank * batch_size + batch_size * it
        end_idx = dp_rank * batch_size + batch_size * (it + 1)
        loaded_data = [self.teacher_logits[idx]]  # Batch size is 1
        arr = np.zeros((batch_size, self.max_seq_length, self.teacher_vocab_size))
        for i in range(len(loaded_data)):
            for j in range(len(loaded_data[i])):
                keys = np.array(list(loaded_data[i][j].keys()), dtype=int)
                values = np.array(list(loaded_data[i][j].values()))
                arr[i, j, keys] = values
                
        logits_tensor = torch.tensor(arr, dtype=torch.bfloat16, device=device)
        return self._shift_tensor_right(logits_tensor, no_model_batch['label'], pad_value=0)
    

    def _compute_white_box_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: Optional[torch.Tensor]):
        student_logits = student_logits[:, :self.max_seq_length, :]
        teacher_probs = teacher_logits[:, :student_logits.size(1), :student_logits.size(-1)]
        mask = (labels != -100).float() if labels is not None else torch.ones_like(student_logits[:, :, 0])
        
        if self.distillation_type == "forward_kld":
            # Forward KLD: student learns from teacher (original implementation)
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='none',
                log_target=False
            ).sum(dim=-1)/torch.sum(mask.view(-1), dim=0) 
        elif self.distillation_type == "reverse_kld":
            # Reverse KLD: teacher provides certainty to student
            loss = F.kl_div(
                torch.log(teacher_probs.clamp(min=1e-10)),  # avoid log(0)
                F.softmax(student_logits, dim=-1),
                reduction='none',
                log_target=False
            ).sum(dim=-1)/torch.sum(mask.view(-1), dim=0) 
        else:
            raise ValueError(f"Unsupported distillation type: {self.distillation_type}. Use 'forward_kld' or 'reverse_kld'")
            
        return (loss * mask).sum() / mask.sum()


    @staticmethod
    def _shift_tensor_right(inputs: torch.Tensor, labels: torch.Tensor, pad_value: float = 0.0):
        batch_size, seqlen, vocab_size = inputs.shape
        device = inputs.device
        labels_ne = labels != -100
        shift_distances = torch.argmax(labels_ne.int(), dim=1)
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch_size, seqlen)
        shifted_idx = idx - shift_distances.unsqueeze(1)
        mask = shifted_idx >= 0
        shifted_idx = shifted_idx.clamp(min=0)
        inputs_flat = inputs.view(batch_size, seqlen, vocab_size)
        shifted_idx = shifted_idx.unsqueeze(2).expand(-1, -1, vocab_size)
        gathered = torch.gather(inputs_flat, 1, shifted_idx)
        mask = mask.unsqueeze(2).expand(-1, -1, vocab_size)
        return torch.where(mask, gathered, torch.full_like(gathered, pad_value))


    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        lm_loss = outputs.loss
        teacher_logits = self._load_teacher_logits(
            batch_size=inputs['input_ids'].size(0),
            it=self.state.global_step,
            dp_rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            device=model.device,
            no_model_batch={'label': inputs.get('labels', None)},
            idx=inputs['idx'].item()
        )
        distil_loss = self._compute_white_box_distillation_loss(
            student_logits=outputs.logits,
            teacher_logits=teacher_logits,
            labels=inputs.get('labels', None)
        )
        total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss
        return (total_loss, outputs) if return_outputs else total_loss