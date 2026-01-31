# Monitor Vast.ai for A100 80GB availability
# Checks every 5 minutes and alerts when available

param(
    [int]$IntervalMinutes = 5,
    [decimal]$MaxPrice = 2.50
)

Write-Host "Monitoring Vast.ai for A100 80GB availability..." -ForegroundColor Cyan
Write-Host "Checking every $IntervalMinutes minutes, max price: `$$MaxPrice/hr" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # Search for A100 80GB
    $result = uv run python -c @"
import subprocess, json
r = subprocess.run(['vastai', 'search', 'offers',
    'gpu_name=A100', 'rentable=true', 'num_gpus>=1', 'gpu_ram>=81920',
    'dph<=$MaxPrice', 'inet_down>=200',
    '--order', 'dph', '--limit', '5', '--raw'],
    capture_output=True, text=True)
if r.returncode == 0 and r.stdout.strip():
    offers = json.loads(r.stdout)
    if offers:
        for o in offers[:3]:
            print(f\"{o['id']}|{o.get('num_gpus',1)}|{o.get('dph_total',0):.2f}|{o.get('gpu_name','')}\")
"@

    if ($result) {
        Write-Host "[$timestamp] " -NoNewline -ForegroundColor Green
        Write-Host "FOUND A100 80GB!" -ForegroundColor Yellow

        foreach ($line in $result -split "`n") {
            if ($line) {
                $parts = $line -split '\|'
                $id = $parts[0]
                $num = $parts[1]
                $price = $parts[2]
                $gpu = $parts[3]
                Write-Host "  Offer $id`: ${num}x $gpu at `$$price/hr" -ForegroundColor Cyan
            }
        }

        Write-Host "`nTo create instance:" -ForegroundColor Yellow
        Write-Host "  uv run bruno-vast create A100_80GB 1`n" -ForegroundColor White

        # Play alert sound
        [Console]::Beep(1000, 500)

        $response = Read-Host "Create instance now? (y/n)"
        if ($response -eq 'y') {
            Write-Host "Creating A100 80GB instance..." -ForegroundColor Green
            uv run bruno-vast create A100_80GB 1
            break
        }
    } else {
        Write-Host "[$timestamp] No A100 80GB available (checking again in $IntervalMinutes min)" -ForegroundColor Gray
    }

    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
