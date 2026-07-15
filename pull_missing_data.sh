#!/bin/bash
# Pull missing T21 cubes from IoA (cap001a) to CSD3
# Run on CSD3. Skips files that already exist.
#
# Usage:
#   1. Copy this script to CSD3: scp pull_missing_data.sh ch2067@login-cpu.hpc.cam.ac.uk:~/
#   2. ssh to CSD3, then:  bash ~/pull_missing_data.sh

set -e

IOA_USER=ch2067
IOA_HOST=cap001a.ast.cam.ac.uk
IOA_BASE=/data/highz6/NewArchives/JitenDhandha/Diffusion_project

DST_BASE=/home/ch2067/rds/hpc-work/ASR21cm/datasets

REDSHIFTS="7 8 9 10 11 12 13"

# === Setup SSH connection multiplexing (only enter password ONCE) ===
mkdir -p ~/.ssh/cm
if ! grep -q "Host ioa-highz" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config <<EOF

Host ioa-highz
    HostName ${IOA_HOST}
    User ${IOA_USER}
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
EOF
    chmod 600 ~/.ssh/config
fi

echo "=== Opening master SSH (enter password if asked) ==="
ssh -fN ioa-highz || true
sleep 2

mkdir -p ${DST_BASE}/varying_IC/T21_cubes
mkdir -p ${DST_BASE}/varying_IC/IC_cubes

# === Pull T21 cubes (varying_IC) for each redshift, skip existing ===
for z in ${REDSHIFTS}; do
    echo ""
    echo "================================================"
    echo "=== varying_IC T21 cubes, z=${z} ==="
    echo "================================================"

    # Count source vs destination
    SRC_COUNT=$(ssh ioa-highz "ls ${IOA_BASE}/varying_IC/T21_cubes/T21_cube_z${z}__Npix256_IC*.mat 2>/dev/null | wc -l")
    DST_COUNT=$(ls ${DST_BASE}/varying_IC/T21_cubes/T21_cube_z${z}__Npix256_IC*.mat 2>/dev/null | wc -l)
    echo "Source: ${SRC_COUNT}  |  Local: ${DST_COUNT}  |  Missing: $((SRC_COUNT - DST_COUNT))"

    if [ "${SRC_COUNT}" -le "${DST_COUNT}" ]; then
        echo "  -> Skipping (already complete)"
        continue
    fi

    rsync -avz --ignore-existing --info=progress2 \
        --include="T21_cube_z${z}__Npix256_IC*.mat" \
        --exclude="*" \
        ioa-highz:${IOA_BASE}/varying_IC/T21_cubes/ \
        ${DST_BASE}/varying_IC/T21_cubes/
done

# === Pull IC cubes (shared across all z) ===
echo ""
echo "================================================"
echo "=== varying_IC IC cubes (delta + vbv) ==="
echo "================================================"
rsync -avz --ignore-existing --info=progress2 \
    --include="delta_Npix256_IC*.mat" \
    --include="vbv_Npix256_IC*.mat" \
    --exclude="*" \
    ioa-highz:${IOA_BASE}/varying_IC/IC_cubes/ \
    ${DST_BASE}/varying_IC/IC_cubes/

# === varying_astro (optional, comment out if not needed) ===
echo ""
echo "================================================"
echo "=== varying_astro T21 cubes ==="
echo "================================================"
mkdir -p ${DST_BASE}/varying_astro/T21_cubes
mkdir -p ${DST_BASE}/varying_astro/parameters

for z in ${REDSHIFTS}; do
    SRC_COUNT=$(ssh ioa-highz "ls ${IOA_BASE}/varying_astro/T21_cubes/T21_cube_z${z}__diffusion_*.mat 2>/dev/null | wc -l")
    DST_COUNT=$(ls ${DST_BASE}/varying_astro/T21_cubes/T21_cube_z${z}__diffusion_*.mat 2>/dev/null | wc -l)
    echo "varying_astro z=${z}: src=${SRC_COUNT} dst=${DST_COUNT}"

    if [ "${SRC_COUNT}" -le "${DST_COUNT}" ]; then continue; fi

    rsync -avz --ignore-existing --info=progress2 \
        --include="T21_cube_z${z}__diffusion_*.mat" \
        --exclude="*" \
        ioa-highz:${IOA_BASE}/varying_astro/T21_cubes/ \
        ${DST_BASE}/varying_astro/T21_cubes/
done

# varying_astro params + shared IC
rsync -avz --ignore-existing --info=progress2 \
    ioa-highz:${IOA_BASE}/varying_astro/parameters/ \
    ${DST_BASE}/varying_astro/parameters/

rsync -avz --ignore-existing --info=progress2 \
    "ioa-highz:${IOA_BASE}/varying_astro/IC_cubes/delta1000.mat" \
    "ioa-highz:${IOA_BASE}/varying_astro/IC_cubes/vbv1000.mat" \
    ${DST_BASE}/varying_astro/IC_cubes/ 2>/dev/null || true

echo ""
echo "=== ALL DONE ==="
echo "Final counts:"
for z in ${REDSHIFTS}; do
    n=$(ls ${DST_BASE}/varying_IC/T21_cubes/T21_cube_z${z}__Npix256_IC*.mat 2>/dev/null | wc -l)
    echo "  z=${z}: ${n} cubes"
done
du -sh ${DST_BASE}/varying_IC ${DST_BASE}/varying_astro
