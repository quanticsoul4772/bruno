#!/bin/bash
# Run this in WSL to download the model with rsync (has resume support)

# Create models directory if it doesn't exist
mkdir -p /mnt/c/Development/Projects/heretic/models

# Download with rsync (will resume if interrupted)
rsync -avz --progress --partial \
  -e "ssh -p 35648 -o ServerAliveInterval=60 -o ServerAliveCountMax=3" \
  root@ssh5.vast.ai:/workspace/models/Qwen2.5-Coder-32B-trial173/ \
  /mnt/c/Development/Projects/heretic/models/Qwen2.5-Coder-32B-trial173/

echo ""
echo "Download complete!"
echo "Location: C:\Development\Projects\heretic\models\Qwen2.5-Coder-32B-trial173"
