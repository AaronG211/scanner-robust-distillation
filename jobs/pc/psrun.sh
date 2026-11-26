#!/bin/zsh
# Run a PowerShell script (stdin) on the PC over ssh: UTF-16LE base64 -> powershell -EncodedCommand.
pre='$ProgressPreference="SilentlyContinue"; $ErrorActionPreference="Continue"; '
script="$pre$(cat)"
enc=$(printf '%s' "$script" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n')
ssh -o ConnectTimeout=20 pc "powershell -NoProfile -NonInteractive -OutputFormat Text -EncodedCommand $enc" 2>&1 | grep -v '^#< CLIXML' | sed '/^<Objs Version/d'
