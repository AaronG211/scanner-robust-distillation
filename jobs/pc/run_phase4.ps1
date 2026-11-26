# Teacher seed variance (MULTI_TEACHER_PROTOCOL.md amendment 2): LoRA seeds 1 and 2 for every teacher, then view-0
# evaluation of all seeds. Waits for run_phase3.ps1 so GPU jobs stay serial. DONE jobs are skipped on re-launch.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_phase4.txt"; $WAITFOR = "$LOG\progress_phase3.txt"
$env:PYTHONUNBUFFERED = "1"
function Say($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $m" | Tee-Object -Append $PROG }
function Run($name, [string[]]$argv) {
  if ((Test-Path $PROG) -and (Select-String -Path $PROG -SimpleMatch "DONE  $name" -Quiet)) { return }
  Say "START $name"
  $p = Start-Process -FilePath $PY -ArgumentList $argv -WorkingDirectory $A -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$LOG\$name.out.txt" -RedirectStandardError "$LOG\$name.err.txt"
  if ($p.ExitCode -ne 0) { Say "FAIL  $name (exit $($p.ExitCode)) - see $LOG\$name.err.txt"; exit 1 }
  Say "DONE  $name"
}
Say "WAIT  until run_phase3.ps1 has finished"
$DONE_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  ALL DONE$'; $FAIL_LINE = '^\d\d-\d\d \d\d:\d\d:\d\d  FAIL '
while (-not ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $DONE_LINE -Quiet))) {
  if ((Test-Path $WAITFOR) -and (Select-String -Path $WAITFOR -Pattern $FAIL_LINE -Quiet)) { Say "FAIL  phase3 failed, not starting"; exit 1 }
  Start-Sleep 60
}
foreach ($fmx in "uni","hopt0","virchow2","uni2h") {
  foreach ($sd in "1","2") {
    foreach ($fd in "XR","S360","CS2") {
      Run "lora_${fmx}_${fd}_s$sd" @("fm_lora.py","--fm",$fmx,"--holdout",$fd,"--variant","plain","--epochs","5","--seed",$sd,"--out","fm_lora_results_seeds.csv")
    }
  }
}
Run "teacher_seeds" @("C:\midog\teacher_seeds.py")
Say "ALL DONE"
