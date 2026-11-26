#!/bin/zsh
# usage: watch.sh <progress file in C:\midog\logs> "<end pattern>"  -- polls every 120 s; tolerates dropped ssh reads
# (gives up after 15 consecutive empty reads = 30 min); stops on the end pattern or FAIL.
empty=0; keep=""; prev=""
while true; do
  txt=$(ssh -o ConnectTimeout=20 pc "type C:\midog\logs\\$1" 2>/dev/null | tr -d '\000\r')
  if [ -z "$txt" ]; then
    empty=$((empty+1)); [ $empty -ge 15 ] && { echo "NO ANSWER FROM PC for 30 min"; break; }
  else
    empty=0; keep="$txt"; last=$(echo "$txt" | tail -1)
    [ "$last" != "$prev" ] && echo "$last" && prev="$last"
    echo "$txt" | grep -q -E "$2|FAIL" && break
  fi
  sleep 120
done
echo "=== full log ==="; echo "$keep"
