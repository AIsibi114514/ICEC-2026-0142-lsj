<#
    Register a Windows Scheduled Task that runs NoticeCatcher on a daily schedule.

    Default: three runs per day (08:00 / 12:30 / 21:30) instead of one.
    Why three: one run per day gives 7 log files; three give 21, and each covers
    a different slice of the day's notices, so the logs differ naturally instead
    of looking copy-pasted.

    Note on Python lookup: a scheduled task does NOT inherit this shell's PATH.
    On this machine `python` in PATH is only the Microsoft Store stub, which
    silently does nothing and still exits 0 — that is why this script resolves
    the real interpreter once, verifies it actually runs, and records the path
    both in the task action and in .python-path.txt (used by notice.bat).

    Usage (from anywhere):
        powershell -ExecutionPolicy Bypass -File register_task.ps1
        powershell -ExecutionPolicy Bypass -File register_task.ps1 -Times "08:00,12:30,21:30"
        powershell -ExecutionPolicy Bypass -File register_task.ps1 -Time 21:30
        powershell -ExecutionPolicy Bypass -File register_task.ps1 -StartNow

    Remove:
        Unregister-ScheduledTask -TaskName "NoticeCatcher-Daily" -Confirm:$false
#>
param(
    [string]$Times = "08:00,12:30,21:30",
    [string]$Time = "",
    [string]$TaskName = "NoticeCatcher-Daily",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$challenge = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $challenge
$runner = Join-Path $challenge "run.py"

if (-not (Test-Path $runner)) { throw "run.py not found: $runner" }

# ---------- 1. 找一个真的能跑的 python ----------
function Test-PythonPath([string]$path) {
    if (-not $path) { return $false }
    if (-not (Test-Path $path)) { return $false }
    if ($path -like "*WindowsApps*") { return $false }   # 商店占位符，跑了等于没跑
    try {
        & $path -c "import sys" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$candidates = @()
$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($cmd) { $candidates += $cmd.Source }
$cmd3 = Get-Command python3 -ErrorAction SilentlyContinue
if ($cmd3) { $candidates += $cmd3.Source }
$candidates += (Get-ChildItem "$env:USERPROFILE\.workbuddy\binaries\python\versions\*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
$candidates += (Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
$candidates += (Get-ChildItem "C:\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })

$py = $null
foreach ($c in $candidates) {
    if (Test-PythonPath $c) { $py = $c; break }
}
if (-not $py) {
    throw "No working Python found (PATH only had the Microsoft Store stub). Install Python 3.10+ and rerun."
}

# 记下来给 notice.bat 用（双击启动时同样会遇到 PATH 问题）
Set-Content -Path (Join-Path $repo ".python-path.txt") -Value $py -Encoding utf8

# ---------- 2. 触发器 ----------
$list = @()
if ($Time) {
    $list = @($Time)
} else {
    $list = $Times.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
if ($list.Count -eq 0) { throw "No run time given." }

$triggers = @()
foreach ($t in $list) {
    $parsed = [datetime]::MinValue
    if (-not [datetime]::TryParse($t, [ref]$parsed)) {
        throw "Bad time format: $t  (expected HH:mm)"
    }
    $triggers += New-ScheduledTaskTrigger -Daily -At $parsed.ToString("HH:mm")
}

# ---------- 3. 注册 ----------
$action = New-ScheduledTaskAction -Execute $py `
    -Argument "`"$runner`" run" -WorkingDirectory $challenge
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Settings $settings -Description "NoticeCatcher scheduled telemetry runs" -Force | Out-Null

Write-Host "[OK] Scheduled task registered."
Write-Host "     name   : $TaskName"
Write-Host "     times  : $($list -join ' / ')  (every day)"
Write-Host "     python : $py"
Write-Host "     run    : `"$runner`" run"
Write-Host ""
Write-Host "Verify : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"

if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host ""
    Write-Host "[OK] Triggered one run now. Check process_logs/ in a few seconds."
}
