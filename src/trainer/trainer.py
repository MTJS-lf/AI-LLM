import torch
from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True, add_dense_layer=False, output_dim=128):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if add_dense_layer:
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=output_dim, bias=False, activation_function=torch.nn.Identity())
        dense_state_dict = torch.load(os.path.join(ckpt_dir, "DenseWeight", "pytorch_model.bin"), map_location=torch.device("cpu"))
        dense_state_dict_new = {"linear.weight": dense_state_dict["weight"]}
        dense_model.load_state_dict(dense_state_dict_new)
    if normlized:
        normlize_layer = models.Normalize()
        if add_dense_layer:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model, normlize_layer], device='cpu')
        else:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        if add_dense_layer:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device='cpu')
        else:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir, safe_serialization=False)


class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode=self.args.sentence_pooling_method,
                                                normlized=self.args.normlized,
                                                add_dense_layer=self.args.add_dense_layer,
                                                output_dim=self.args.embedding_dim)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class BaseTrainer(Trainer):
   # def _save(self, output_dir: Optional[str] = None, state_dict=None):
   #     output_dir = output_dir if output_dir is not None else self.args.output_dir
   #     os.makedirs(output_dir, exist_ok=True)
   #     logger.info("Saving model checkpoint to %s", output_dir)
   #     if not hasattr(self.model, 'save_pretrained'):
   #         raise NotImplementedError(f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
   #     else:
   #         self.model.save_pretrained(output_dir)
   #     if self.tokenizer is not None and self.is_world_process_zero():
   #         self.tokenizer.save_pretrained(output_dir)

   #     torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(**inputs)
        print("loss=", output)
        loss = output["loss"]
        logits = output["logits"]
        return (loss, output) if return_outputs else loss


class STETrainer(Trainer):
    def __init__(self, efficient_save, **kwargs):
        super().__init__(**kwargs)
        self.efficient_save = efficient_save

    def save_ckpt_for_sentence_transformers(self, tmp_dir, output_dir, pooling_mode: str = 'mean'):
        '''convert to sentence transformer format'''
        import shutil
        from sentence_transformers import models, SentenceTransformer
        word_embedding_model = models.Transformer(tmp_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        if os.path.exists(os.path.join(tmp_dir, 'scaling_layer.bin')):
            state_dict = torch.load(os.path.join(tmp_dir, 'scaling_layer.bin'))
            in_features, out_features = state_dict['linear.weight'].shape[1], state_dict['linear.weight'].shape[0]
            scaling_layer = models.Dense(in_features, out_features, bias=True, activation_function=torch.nn.modules.linear.Identity())
            scaling_layer.load_state_dict(state_dict, strict=True)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, scaling_layer], device='cpu')
        else:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
        model.save(output_dir, safe_serialization=False)
        shutil.rmtree(tmp_dir)

    def _save(self, output_dir: Optional[str] = None, **kwargs):
        '''save the unwrap model'''

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_dir)
        unwrap_model = self.model.embedder.encoder
        if self.is_world_process_zero():
            # first saves to the tmp dir, then converts to sentence-transformer
            tmp_dir = output_dir + '-tmp'
            unwrap_model.save_pretrained(tmp_dir, safe_serialization=self.args.save_safetensors)
            self.tokenizer.save_pretrained(tmp_dir)
            if hasattr(self.model, 'scaling_layer'):
                scaling_layer = {'linear.weight': self.model.scaling_layer.state_dict()['linear.weight'].data.cpu(), 
                                    'linear.bias': self.model.scaling_layer.state_dict()['linear.bias'].data.cpu()}
                torch.save(scaling_layer, os.path.join(tmp_dir, 'scaling_layer.bin'))
            self.save_ckpt_for_sentence_transformers(tmp_dir, output_dir, self.model.embedder.pooling_strategy.value)

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.efficient_save:
            '''only save the model ckpt weights to save disk mem'''
            from transformers.trainer import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
        else:
            super()._save_checkpoint(model, trial, metrics)

class NerTrainer(Trainer):
    def __init__(self, crf_layer_lr=0.1, **kwargs):
        super().__init__(**kwargs)
        self.crf_layer_lr = crf_layer_lr

    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(**inputs)
        loss = output["loss"]
        logits = output["logits"]
        return (loss, output) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            crf_parameters = self.get_crf_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in crf_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in crf_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.crf_layer_lr
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
            

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def get_crf_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        all_model_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        crf_layer_parameters = [name for name in all_model_parameters if "crf" in name]
        return crf_layer_parameters
    

