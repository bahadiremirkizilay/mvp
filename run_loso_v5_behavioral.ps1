param([int]$StartFold = 1, [int]$EndFold = 120)
Set-Location "c:\Users\furka\Desktop\projects\mvp"
for ($fold = $StartFold; $fold -le $EndFold; $fold++) {
    $foldDir = "checkpoints\deception_loso_v5_behavioral\fold_{0:D3}" -f $fold
    $metricsFile = "$foldDir\metrics.json"
    if (Test-Path $metricsFile) {
        Write-Host "Skipping fold $fold (already done)"
        continue
    }
    New-Item -ItemType Directory -Force -Path $foldDir | Out-Null
    Write-Host "=== Running fold $fold ==="
    python fusion/train_fusion.py `
        --dataset reallife_2016 `
        --manifest "data\RealLifeDeceptionDetection.2016\deception_manifest.csv" `
        --feature_cache_dir checkpoints/feature_cache_v5 `
        --feature_mode real `
        --cache_version v5 `
        --modalities behavioral `
        --loso_fold $fold `
        --num_epochs 30 `
        --batch_size 16 `
        --lr 1e-3 `
        --hidden_dim 96 `
        --num_layers 1 `
        --dropout 0.3 `
        --weight_decay 5e-4 `
        --output_dir $foldDir
}
Write-Host "=== All folds completed ==="
