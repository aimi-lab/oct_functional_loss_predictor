
import torch

class LoadMixin():
    def from_checkpoint_path(self, checkpoint_path: str, state_dict_key: str = None, **kwargs):
        """
        Utility function to load a model from a checkpoint path. If the target model is a sub-model in the checkpoint,
        it can be loaded by specifying the state_dict_key.

        Args:
            checkpoint_path (str): Path to the checkpoint.
            state_dict_key (str): Key to the sub-model in the checkpoint. 
                E.g Checkpoint holds keys: "model.encoder.layer1" "model.encoder.layer2", ... , 
                "model.decoder.layer1", "model.decoder.layer2", ...,. 
                To load the encoder, specify state_dict_key="model.encoder".
        """
        # Checkpoints are loaded to the same device as they were saved. 
        # Map the checkpoint to the current device if needed.
        device = self._get_model_device()

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        state_dict = checkpoint["state_dict"]

        if state_dict_key is not None:
            state_dict = self._extract_sub_statedict(state_dict, state_dict_key)

        self.load_state_dict(state_dict, strict=False)

    def _get_model_device(self):
        return next(self.parameters()).device
    
    @staticmethod
    def _extract_sub_statedict(statedict: dict, state_dict_key: str)-> dict:

        state_dict_key = state_dict_key.replace(" ", "")
        if not state_dict_key.endswith("."):
            state_dict_key += "."

        sub_statedict = {}

        for key, value in statedict.items():
            if key.startswith(state_dict_key):
                key = key.replace(state_dict_key, "")
                sub_statedict[key] = value

        return sub_statedict
    