# Experiment E1.7 (MULTI_TEACHER_PROTOCOL.md): Virchow2, UNI, UNI2-h teachers -> ResNet-18 KD and KD+TGA students.
# One teacher's full chain at a time, so results arrive teacher by teacher. DONE jobs are skipped on re-launch.
# NOTE: PowerShell variable names are case-insensitive, so every loop variable has its own distinct name.
$ErrorActionPreference = "Stop"
$PY = "C:\midog\venv\Scripts\python.exe"
$A = "C:\midog\repo\axel_model"
$LOG = "C:\midog\logs"; $PROG = "$LOG\progress_e17.txt"
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
$fmList = "virchow2","uni","uni2h"; $foldList = "XR","S360","CS2"; $halfList = "A","B"
$seeds = @("--seeds","0","1","2","3","4","5","6","7","8","9")
$student = @("--alpha","0","--T","2","--beta","1","--backbone","resnet18","--epochs","60","--save","--out","resnet18_distill_e17.csv") + $seeds
foreach ($fmx in $fmList) {
  foreach ($fd in $foldList) { Run "lora_${fmx}_$fd" @("fm_lora.py","--fm",$fmx,"--holdout",$fd,"--variant","plain","--epochs","5","--seed","0") }
  foreach ($fd in $foldList) {
    Run "cache_${fmx}_$fd" @("distill_teacher.py","--fm",$fmx,"--holdout",$fd)
    Run "cacheall_${fmx}_$fd" @("distill_teacher.py","--fm",$fmx,"--holdout",$fd,"--target_all")
  }
  Run "teacher_eval_$fmx" @("C:\midog\teacher_eval.py")
  foreach ($fd in $foldList) {
    Run "kd_${fmx}_$fd" (@("distill.py","--holdout",$fd,"--tag","r18_kd_$fmx","--teacher_fm",$fmx) + $student)
    foreach ($hv in $halfList) {
      Run "tga_${fmx}_${fd}_$hv" (@("distill.py","--holdout",$fd,"--tag","r18_kd_tga_$fmx$hv","--teacher_fm",$fmx,"--target_half",$hv) + $student)
    }
  }
  Run "eval_$fmx" @("r18_crossfit_eval.py","--arms","r18_kd_$fmx","r18_kd_tga_$fmx","--out","r18_e17_$fmx")
}
Say "ALL DONE"
