# Viewer's README
## How to start?
```bash
MLPSP_TILE_ORDER=tilecenter python viewer/run_viewer.py --output_dir /home/sjkim/Documents/mlpsp/output/mip360/screenshots --ckpt ~/Desktop/room.pt --engine tile_cuda --data_dir /media/sjkim/HDD_3TB/dataset/360_v2/room/
```

## VLM query viewer
```bash
MLPSP_TILE_ORDER=tilecenter python viewer/run_viewer_vlm.py \
  --output_dir /home/sjkim/Documents/mlpsp/output/mip360/screenshots \
  --ckpt ~/Desktop/room.pt \
  --engine tile_cuda \
  --data_dir /media/sjkim/HDD_3TB/dataset/360_v2/room/ \
  --decoder_ckpt /path/to/decoder_epoch_02000.pt \
  --vlm_edit_config viewer/configs/vlm_edit_labels.yaml
```

Notes:
- `--decoder_ckpt` is required when primitive semantic dim and text embedding dim differ.
- Candidate labels are loaded from `--vlm_edit_config` and selected from a dropdown in GUI.
- Query operation uses multi-prompt softmax when multiple labels are configured.
- Applying extraction/deletion overwrites the previous query mask (non-cumulative apply).
- The viewer keeps original mouse selection tools and combines them with query-based operations.
