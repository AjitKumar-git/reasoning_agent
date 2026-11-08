#!/usr/bin/env bash
# ==============================================================================
# CerebroSpheroid V4: GCP Cloud TPU VM Automation Script
# ==============================================================================
set -e

TPU_NAME="${TPU_NAME:-cerebro-v4-tpu}"
ZONE="${ZONE:-us-central1-a}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-1}" # v5litepod-1 (v5e 1 chip), v3-8 (v3 8 cores), or v5litepod-4
SOFTWARE_VERSION="${SOFTWARE_VERSION:-v2-alpha-tpuv5-lite}" # or tpu-ubuntu2204-base for v3-8
USE_SPOT="${USE_SPOT:-true}"

function show_help() {
    echo "Usage: $0 [create|ssh|run|status|stop|delete]"
    echo ""
    echo "Environment Variables (optional):"
    echo "  PROJECT_ID          GCP Project ID (default: current gcloud project)"
    echo "  TPU_NAME            Name of TPU VM (default: cerebro-v4-tpu)"
    echo "  ZONE                GCP Zone (default: us-central1-a)"
    echo "  ACCELERATOR_TYPE    v5litepod-1 (v5e), v5litepod-4, or v3-8 (default: v5litepod-1)"
    echo "  USE_SPOT            true for ~70% spot discount, false for on-demand (default: true)"
    echo ""
    echo "Commands:"
    echo "  login               Authenticate gcloud with your Google account"
    echo "  create              Provision Cloud TPU VM on GCP"
    echo "  ssh                 Interactive SSH into the TPU VM"
    echo "  run                 Bootstrap environment & launch CerebroSpheroid V4 on TPU VM"
    echo "  status              Check TPU VM status"
    echo "  stop                Stop the TPU VM"
    echo "  delete              Delete the TPU VM to prevent any billing"
}

ACTION="${1:-help}"

case "$ACTION" in
    login)
        echo "🔐 Authenticating gcloud..."
        gcloud auth login --update-adc
        ;;
    create)
        echo "🚀 Provisioning Cloud TPU VM '$TPU_NAME' in zone '$ZONE' ($ACCELERATOR_TYPE)..."
        SPOT_FLAG=""
        if [ "$USE_SPOT" = "true" ]; then
            SPOT_FLAG="--spot"
            echo "💡 Using SPOT instance for maximum cost savings."
        fi
        
        # Select appropriate software version
        if [[ "$ACCELERATOR_TYPE" =~ v3-.* ]]; then
            SOFTWARE_VERSION="tpu-ubuntu2204-base"
        elif [[ "$ACCELERATOR_TYPE" =~ v5.* ]]; then
            SOFTWARE_VERSION="v2-alpha-tpuv5-lite"
        fi

        gcloud compute tpus tpu-vm create "$TPU_NAME" \
            --zone="$ZONE" \
            --accelerator-type="$ACCELERATOR_TYPE" \
            --version="$SOFTWARE_VERSION" \
            $SPOT_FLAG

        echo "✅ TPU VM '$TPU_NAME' created successfully!"
        ;;
    ssh)
        echo "🔗 Connecting to TPU VM '$TPU_NAME'..."
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE"
        ;;
    run)
        echo "🚀 Bootstrapping and launching training on TPU VM '$TPU_NAME'..."
        # Run startup and launch script remotely via gcloud ssh
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="
            set -e
            echo '📦 Preparing environment...'
            sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip
            if [ ! -d reasoning_agent ]; then
                git clone https://github.com/AjitKumar-git/reasoning_agent.git
            fi
            cd reasoning_agent
            git fetch origin main && git reset --hard origin/main
            pip install -q -U 'datasets>=3.0.0' pyarrow transformers sentencepiece optax 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html torch
            mkdir -p /tmp/checkpoints
            echo '🚀 Launching train_v4_jax.py...'
            nohup python3 training/train_v4_jax.py \
                --ckpt_dir /tmp/checkpoints \
                --resume_target latest \
                --dataset_source dense_edu \
                --elastic_window \
                --total_window 1024 \
                --batch_size 16 \
                --grad_accum 4 \
                --lr 0.0005 \
                --deliberation_mode scheduled \
                --schedule_steps 20000 \
                --min_sampling_p 0.35 \
                --repetition_penalty 1.25 \
                --eval_interval 200 \
                --thalamic_anchor_weight 0.1 \
                --max_hours 8.5 \
                --cloud_save_hours 1.0 \
                --log_every 10 \
                --cloud_sync \
                --source_node gcp_tpu > training.log 2>&1 &
            echo '✅ Training started in background (PID: \$!). Logs: tail -f ~/reasoning_agent/training.log'
        "
        ;;
    status)
        gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE"
        ;;
    stop)
        echo "🛑 Stopping TPU VM '$TPU_NAME'..."
        gcloud compute tpus tpu-vm stop "$TPU_NAME" --zone="$ZONE"
        ;;
    delete)
        echo "🗑️ Deleting TPU VM '$TPU_NAME'..."
        gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet
        echo "✅ TPU VM deleted."
        ;;
    *)
        show_help
        ;;
esac
