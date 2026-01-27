# RunPod Automation Pain Points & Resolutions

This document identifies pain points discovered during the heretic abliteration workflow automation and their planned resolutions.

## Critical Pain Points

### 1. SSH Key Not Recognized on New Pods
**Problem:** When creating a new pod, SSH authentication fails with "Permission denied (publickey)" even though the SSH key exists locally.

**Root Cause:** The SSH public key must be registered in RunPod account settings (Settings → SSH Keys) BEFORE creating pods. The `SSH_PUBLIC_KEY` environment variable only works for pods that already have SSH configured.

**Resolution:**
1. User must manually add their SSH public key to RunPod account settings once
2. Script now includes the SSH key via `env` array during pod creation as a backup
3. Added `check-tools` command that verifies SSH key exists

**Action Required:** Add your SSH public key to RunPod:
```
1. Go to https://runpod.io/console/user/settings
2. Scroll to "SSH Public Keys"
3. Paste contents of: ~/.ssh/id_ed25519.pub
4. Save
```

### 2. Heretic Crashes with EOFError in Non-Interactive Mode
**Problem:** When running heretic via `nohup` or background process, it crashes with `EOFError` from `prompt_toolkit` because there's no TTY for interactive prompts.

**Root Cause:** Heretic uses `questionary` for interactive trial selection which requires a terminal.

**Resolution:** ✅ FIXED - Added `--auto-select` flag to heretic:
```bash
# Auto-select best trial and save to default path
heretic --auto-select Qwen/Qwen3-4B-Instruct-2507

# Auto-select and save to specific path
heretic --auto-select --auto-select-path /workspace/models/output Qwen/Qwen3-4B-Instruct-2507
```

Alternative workarounds:
1. Use `script` command to provide pseudo-TTY: `script -q -c "heretic model" /workspace/heretic.log`
2. Run heretic interactively in an SSH session (user must be present)

### 3. Cannot See Heretic Output When Running via SSH Heredoc
**Problem:** When launching heretic through automated SSH, the output isn't visible and we can't tell what state it's in.

**Root Cause:** Output goes to the remote terminal, not captured locally.

**Resolutions:**
1. Always redirect output to log file: `heretic model > /workspace/heretic.log 2>&1`
2. Added `progress` command to check process status and log files
3. Use `tail -f /workspace/heretic.log` to monitor in real-time

### 4. Direct TCP SSH Connection Often Refused
**Problem:** The direct TCP SSH connection (e.g., `ssh root@IP -p PORT`) frequently shows "Connection refused" even when pod is RUNNING.

**Root Cause:** RunPod's direct TCP SSH is unreliable; the SSH proxy is more stable.

**Resolution:** Always prefer SSH proxy (`podId-userId@ssh.runpod.io`) over direct TCP.

### 5. SSH Proxy Address Uses Wrong User ID Format
**Problem:** The GraphQL API's `myself.id` returns a long string like `user_31LKZHqPR9mDqL5r7JAc2TfI1Ui`, but the SSH proxy requires a numeric ID like `64411784`.

**Root Cause:** RunPod has two different user ID formats:
- **API format:** `user_XXXXX...` (returned by `myself { id }` query)
- **SSH format:** `64411784` (numeric, shown in RunPod console SSH connection strings)

**Resolution:**
1. Store the numeric `RUNPOD_USER_ID` as a constant in the script
2. Find this ID from RunPod console → Pod Details → SSH connection string
3. The pod ID changes with each pod, but the user ID is constant for your account

**SSH Proxy Format:**
```
{podId}-{userId}@ssh.runpod.io

Example: znwwcgs2lwcra3-64411784@ssh.runpod.io
         ^^^^^^^^^^^^^^  ^^^^^^^^
         Pod ID (changes) User ID (constant)
```

### 6. Script Creates New Pod Instead of Restarting Existing One
**Problem:** When a pod is stopped/exited, the script created a new pod instead of restarting the existing one, wasting money.

**Root Cause:** Poor error handling in start-pod logic.

**Resolution:** 
1. Try `start-pod` first for stopped/exited pods
2. Only create new pod if no existing pods or all terminated
3. Added better pod status checking

## Medium Priority Pain Points

### 7. PowerShell Parser Error with `<` Character
**Problem:** Heredoc commands containing `<` character cause PowerShell parser errors.

**Resolution:** Escape or avoid `<` in heredoc strings, use `[model-name]` instead of `<model-name>`.

### 8. Script Config Not Auto-Updating Properly
**Problem:** `Update-ScriptConfig` fails with "Cannot overwrite variable Host" error.

**Root Cause:** `$Host` is a reserved automatic variable in PowerShell.

**Resolution:** Renamed parameter from `-Host` to `-HostAddress`. ✅ FIXED

### 9. WSL SSH Key Permissions
**Problem:** SSH key copied to WSL from Windows has wrong permissions (0777).

**Resolution:** Script now includes `chmod 600` after copying key to WSL.

## Workflow Improvements Needed

### 10. Add `run-abliterate` Command
Create a single command that:
1. Checks pod is running (or starts it)
2. Runs setup if needed
3. Starts heretic with `script` for TTY
4. Monitors progress
5. Handles trial selection (future: auto-select)

### 11. Add Progress Monitoring
- Continuously poll and display:
  - Process status (running/sleeping/stopped)
  - GPU utilization
  - Log file tail
  - Estimated time remaining

### 12. Better Error Recovery
- Detect when heretic crashes
- Auto-restart with different parameters
- Save partial progress

## Quick Reference: Working SSH Commands

```powershell
# Via WSL with heredoc (WORKS for commands)
powershell -Command "wsl -e bash -c 'ssh -tt -o StrictHostKeyChecking=no podId-userId@ssh.runpod.io <<SSHEOF
commands here
exit
SSHEOF'"

# Interactive SSH (for monitoring)
wsl bash -c "ssh -tt podId-userId@ssh.runpod.io"
```

## Status

| Pain Point | Status | Priority |
|------------|--------|----------|
| SSH Key Recognition | Documented workaround | Critical |
| SSH Proxy User ID Format | Fixed - use constant | Critical |
| EOFError Non-Interactive | Fixed - `--auto-select` flag | Critical |
| Output Visibility | Log file + progress cmd | High |
| Direct TCP Refused | Use SSH proxy | Medium |
| Create vs Restart | Logic improved | Medium |
| PowerShell `<` Error | Fixed | Low |
| Config Update Error | Fixed | Low |
| WSL Key Permissions | Fixed | Low |
