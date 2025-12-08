#!/bin/bash
# Demo script to generate captions for 6 diverse images
# For video demonstration purposes

CHECKPOINT="checkpoints/best_model_real.pt"
TOKENIZER="checkpoints/tokenizer.json"

# Color codes for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo "                    OpenVision 2 - Demo: 6 Images                            "
echo "=============================================================================="
echo ""

# Array of 6 diverse demo images
declare -a IMAGES=(
    "data/Images/1000268201_693b08cb0e.jpg"  # Child in pink dress
    "data/Images/1001773457_577c3a7d70.jpg"  # Two dogs
    "data/Images/1002674143_1b742ab4b8.jpg"  # Girl painting
    "data/Images/1003163366_44323f5815.jpg"  # Man on bench with dog
    "data/Images/1007129816_e794419615.jpg"  # Man with orange hat
    "data/Images/1007320043_627395c3d8.jpg"  # Child climbing rope
)

declare -a DESCRIPTIONS=(
    "Child in Pink Dress (Children/People)"
    "Two Dogs Interacting (Animals)"
    "Girl Painting Rainbow (Activities/Art)"
    "Man on Bench with Dog (People + Animals)"
    "Man with Orange Hat (Portrait)"
    "Child Climbing Rope (Action/Sports)"
)

# Loop through all 6 images
for i in "${!IMAGES[@]}"; do
    IMAGE="${IMAGES[$i]}"
    DESC="${DESCRIPTIONS[$i]}"
    NUM=$((i + 1))

    echo ""
    echo "=============================================================================="
    echo -e "${BLUE}Image $NUM/6: $DESC${NC}"
    echo "Path: $IMAGE"
    echo "------------------------------------------------------------------------------"

    # Check if image exists
    if [ ! -f "$IMAGE" ]; then
        echo -e "${YELLOW}⚠️  Image not found, skipping...${NC}"
        continue
    fi

    # Generate caption
    echo -e "${GREEN}Generating caption...${NC}"
    python src/inference_real.py \
        --checkpoint "$CHECKPOINT" \
        --tokenizer "$TOKENIZER" \
        --image "$IMAGE" \
        --max-length 50

    echo ""

    # Pause between images (optional - remove for continuous)
    # sleep 2
done

echo ""
echo "=============================================================================="
echo "✓ Demo complete! All 6 captions generated."
echo "=============================================================================="
