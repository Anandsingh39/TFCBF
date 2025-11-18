python -m src.models.modules.a \
  --data src/models/modules/pact_dataset_ctx16.npz \
  --state-key state_ctx \
  --action-key action_ctx \
  --seq-len 16 \
  --batch-size 8 \
  --ordered \
  --seed 123
