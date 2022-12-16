from fairseq.criterions.multitask_crossentropy_with_contrastive_with_extra_mt import MultiTaskCrossEntropyWithContrastiveWithExtraMT
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import torch
from fairseq import metrics, utils

@register_criterion('multi_task_cross_entropy_with_contrastive_with_extra_mt_rep')
class MultiTaskCrossEntropyWithContrastiveWithExtraMTRep(MultiTaskCrossEntropyWithContrastiveWithExtraMT):
    def compute_loss_asr(self, model, sample, reduce=True):
        net_output, _ = model(sample["net_input"]["src_tokens"],
                              sample["net_input"]["src_lengths"],
                              sample["prev_output_src_tokens"],
                              task="asr")
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["source"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
