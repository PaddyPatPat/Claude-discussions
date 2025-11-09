# TrueNAS Mac Rsync SSH Setup

## Overview

This guide explains how to set up secure, password-less authentication between a macOS system and TrueNAS for Rsync operations using SSH key pairs. This allows for automated backups and file synchronization without storing passwords.

## Prerequisites

- Mac with SSH client (built-in on macOS)
- TrueNAS server with network access
- Admin access to TrueNAS web interface
- Network connectivity between Mac and TrueNAS

## SSH Key-Based Authentication

SSH keys provide secure authentication without passwords by using asymmetric cryptography:

- **Private key**: Stays on your Mac (~/.ssh/id_rsa), never shared
- **Public key**: Copied to TrueNAS (~/.ssh/id_rsa.pub), safe to share

When you connect, TrueNAS verifies you possess the private key matching its stored public key.

## Setup Steps

### Step 1: Generate SSH Key Pair on Mac

If you don't already have an SSH key pair, generate one:

```bash
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```

**During key generation**:

1. **Save location prompt**: Press Enter to use default (`~/.ssh/id_rsa`)
   ```
   Enter file in which to save the key (/Users/username/.ssh/id_rsa):
   ```

2. **Passphrase prompt**: Choose whether to add extra security
   ```
   Enter passphrase (empty for no passphrase):
   ```
   - **With passphrase**: More secure, but you'll need to enter it (can use ssh-agent to cache)
   - **Without passphrase**: Convenient for automation, less secure if Mac is compromised

3. **Confirm passphrase**: Re-enter if you set one
   ```
   Enter same passphrase again:
   ```

**Key generation output**:
```
Your identification has been saved in /Users/username/.ssh/id_rsa
Your public key has been saved in /Users/username/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:abcdef123456... your-email@example.com
```

**Generated files**:
- `~/.ssh/id_rsa` - Private key (keep secret)
- `~/.ssh/id_rsa.pub` - Public key (share with TrueNAS)

### Step 2: Copy Your Public Key

Display your public key:

```bash
cat ~/.ssh/id_rsa.pub
```

**Example output**:
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC7... your-email@example.com
```

Copy the **entire output** (starts with `ssh-rsa`, ends with your email).

**Alternative**: Copy directly to clipboard:

```bash
# macOS
pbcopy < ~/.ssh/id_rsa.pub

# Or install xclip
brew install xclip
xclip -sel clip < ~/.ssh/id_rsa.pub
```

### Step 3: Configure TrueNAS User

Access the TrueNAS web interface and add your SSH public key:

#### Option A: Use Existing User

1. Navigate to **Accounts** → **Users**
2. Click on the user you want to use for Rsync (e.g., `admin`, `backup-user`)
3. Click **Edit** (pencil icon)
4. Scroll down to **SSH Public Key** field
5. Paste your public key (entire output from Step 2)
6. Click **Save**

#### Option B: Create New User

1. Navigate to **Accounts** → **Users**
2. Click **Add**
3. Fill in required fields:
   - **Username**: `rsync-user` (or your preferred name)
   - **Full Name**: Descriptive name
   - **Email**: Your email
   - **Password**: Set a strong password (fallback authentication)
   - **User ID**: Auto-assigned or specify
   - **Primary Group**: Create new or use existing
4. Set **Home Directory**: `/mnt/pool/users/rsync-user`
5. Paste your public key in **SSH Public Key** field
6. **Shell**: `/usr/bin/bash` or `/usr/bin/sh`
7. Click **Save**

#### Set Appropriate Permissions

Ensure the user has access to directories you'll sync:

1. Navigate to **Storage** → **Pools**
2. Find the dataset/directory for Rsync
3. Click the three dots → **Edit Permissions**
4. Add ACL entry or set UNIX permissions for your user:
   - **User**: Your rsync user
   - **Permissions**: Read/Write or as needed
5. Apply recursively if needed
6. Click **Save**

### Step 4: Enable SSH Service on TrueNAS

1. Navigate to **Services**
2. Find **SSH** in the service list
3. Toggle the switch to **enable** SSH
4. Click the **gear icon** to configure SSH settings:

**Recommended SSH settings**:

- **TCP Port**: `22` (default) or custom port for security
- **Log in as Root with Password**: **Disabled** (security best practice)
- **Allow Password Authentication**: **Disabled** after verifying key works (force key-only)
- **Allow Kerberos Authentication**: Disabled (unless needed)
- **Allow TCP Port Forwarding**: As needed
- **Compress Connections**: Enabled (improves Rsync performance)
- **SFTP Log Level**: INFO (for debugging)
- **SFTP Log Facility**: USER
- **Weak Ciphers**: Disabled (security)

5. Click **Save**
6. Ensure SSH service is **Running** (green checkmark)

### Step 5: Test SSH Connection

From your Mac terminal, test the connection:

```bash
ssh username@truenas-ip-address
```

**Example**:
```bash
ssh admin@192.168.1.100
```

**First connection**:
```
The authenticity of host '192.168.1.100 (192.168.1.100)' can't be established.
ED25519 key fingerprint is SHA256:xyz123...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type `yes` and press Enter. This adds TrueNAS to your known hosts.

**Successful connection**:
- You should connect without password prompt
- You'll see TrueNAS shell prompt

**If prompted for password**:
- SSH key authentication failed
- Verify public key was pasted correctly on TrueNAS
- Check file permissions (see Troubleshooting section)

**Exit the session**:
```bash
exit
```

## SSH Config File for Simplified Access

Create or edit `~/.ssh/config` to simplify connections:

### Create Config File

If `~/.ssh/config` doesn't exist:

```bash
touch ~/.ssh/config
chmod 600 ~/.ssh/config
```

**Important**: The config file must have restrictive permissions (600) for security.

### Basic Configuration

Add this entry to `~/.ssh/config`:

```
Host truenas
    HostName 192.168.1.100
    User admin
    Port 22
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
```

**Configuration breakdown**:

| Option | Description | Example |
|--------|-------------|---------|
| `Host` | Alias for connection (use any name) | `truenas`, `nas`, `backup-server` |
| `HostName` | Actual IP address or DNS name | `192.168.1.100`, `truenas.local` |
| `User` | Username on TrueNAS | `admin`, `rsync-user` |
| `Port` | SSH port | `22` (default), `2222` (custom) |
| `IdentityFile` | Path to private key | `~/.ssh/id_rsa`, `~/.ssh/truenas_rsa` |
| `IdentitiesOnly` | Only try specified key | `yes` (recommended) |

### Enhanced Configuration

Add these options for improved reliability and performance:

```
Host truenas
    HostName 192.168.1.100
    User admin
    Port 22
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    Compression yes
    TCPKeepAlive yes
    StrictHostKeyChecking accept-new
    LogLevel ERROR
```

**Additional options explained**:

| Option | Purpose | Value |
|--------|---------|-------|
| `ServerAliveInterval` | Send keepalive every N seconds | `60` (prevents timeout) |
| `ServerAliveCountMax` | Max failed keepalives before disconnect | `3` (180 seconds total) |
| `Compression` | Compress data over SSH tunnel | `yes` (faster on slow networks) |
| `TCPKeepAlive` | TCP-level keepalive | `yes` (detect dead connections) |
| `StrictHostKeyChecking` | Host key verification | `accept-new` (auto-accept on first connect) |
| `LogLevel` | SSH client logging | `ERROR` (quiet), `INFO` (verbose) |

### Multiple TrueNAS Servers

Configure multiple servers with different aliases:

```
Host truenas-home
    HostName 192.168.1.100
    User admin
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host truenas-office
    HostName 10.0.1.50
    User backup-user
    Port 2222
    IdentityFile ~/.ssh/truenas_office_rsa

Host truenas-remote
    HostName nas.example.com
    User rsync
    Port 22
    IdentityFile ~/.ssh/truenas_remote_rsa
    Compression yes
```

### Usage After Configuration

Once configured, connections become much simpler:

**SSH connection**:
```bash
# Before
ssh admin@192.168.1.100 -p 22 -i ~/.ssh/id_rsa

# After
ssh truenas
```

**Rsync**:
```bash
# Before
rsync -avz -e "ssh -p 22 -i ~/.ssh/id_rsa" /local/path/ admin@192.168.1.100:/remote/path/

# After
rsync -avz /local/path/ truenas:/remote/path/
```

**SCP file copy**:
```bash
# Before
scp -P 22 -i ~/.ssh/id_rsa file.txt admin@192.168.1.100:/mnt/pool/backup/

# After
scp file.txt truenas:/mnt/pool/backup/
```

## Rsync Usage Examples

### Basic Rsync Commands

**Sync local to TrueNAS**:
```bash
rsync -avz /Users/username/Documents/ truenas:/mnt/pool/backup/documents/
```

**Sync TrueNAS to local**:
```bash
rsync -avz truenas:/mnt/pool/backup/documents/ /Users/username/Documents/
```

**Common Rsync options**:

| Option | Description |
|--------|-------------|
| `-a` | Archive mode (recursive, preserve permissions, timestamps, symlinks) |
| `-v` | Verbose output |
| `-z` | Compress during transfer |
| `-h` | Human-readable numbers |
| `-P` | Show progress and keep partial transfers |
| `--delete` | Delete files on destination that don't exist on source |
| `--dry-run` | Simulate transfer without making changes |
| `--exclude` | Exclude files/directories |

### Advanced Rsync Examples

**Dry run (preview changes)**:
```bash
rsync -avzn /local/path/ truenas:/remote/path/
# or
rsync -avz --dry-run /local/path/ truenas:/remote/path/
```

**With progress and human-readable sizes**:
```bash
rsync -avzhP /local/path/ truenas:/remote/path/
```

**Delete files on destination not in source** (mirror):
```bash
rsync -avz --delete /local/path/ truenas:/remote/path/
```

**Exclude specific files/directories**:
```bash
rsync -avz --exclude '.DS_Store' --exclude 'node_modules/' \
    /Users/username/Projects/ truenas:/mnt/pool/backup/projects/
```

**Bandwidth limit** (useful for background syncs):
```bash
rsync -avz --bwlimit=5000 /local/path/ truenas:/remote/path/
# Limits to 5000 KB/s (5 MB/s)
```

**Backup with timestamp** (creates dated directory):
```bash
BACKUP_DATE=$(date +%Y-%m-%d_%H-%M-%S)
rsync -avz /Users/username/Documents/ \
    truenas:/mnt/pool/backup/documents-$BACKUP_DATE/
```

**Incremental backup with hard links** (saves space):
```bash
BACKUP_DATE=$(date +%Y-%m-%d)
LATEST=/mnt/pool/backup/latest

rsync -avz --delete \
    --link-dest=$LATEST \
    /Users/username/Documents/ \
    truenas:/mnt/pool/backup/documents-$BACKUP_DATE/

# Update latest symlink
ssh truenas "ln -nsf /mnt/pool/backup/documents-$BACKUP_DATE $LATEST"
```

### Automated Backup Script

Create a backup script (`~/bin/backup-to-truenas.sh`):

```bash
#!/bin/bash

# Configuration
SOURCE_DIR="/Users/username/Documents"
DEST_HOST="truenas"
DEST_DIR="/mnt/pool/backup/documents"
LOG_FILE="$HOME/rsync_backup.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Rsync options
RSYNC_OPTS="-avzhP --delete --exclude='.DS_Store' --exclude='*.tmp'"

# Run backup
echo "[$TIMESTAMP] Starting backup..." | tee -a "$LOG_FILE"

rsync $RSYNC_OPTS "$SOURCE_DIR/" "$DEST_HOST:$DEST_DIR/" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] Backup completed successfully" | tee -a "$LOG_FILE"
else
    echo "[$TIMESTAMP] Backup failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
```

**Make script executable**:
```bash
chmod +x ~/bin/backup-to-truenas.sh
```

**Run manually**:
```bash
~/bin/backup-to-truenas.sh
```

**Automate with cron** (add to `crontab -e`):
```
# Daily backup at 2 AM
0 2 * * * /Users/username/bin/backup-to-truenas.sh

# Every 6 hours
0 */6 * * * /Users/username/bin/backup-to-truenas.sh
```

**Automate with launchd** (macOS preferred over cron):

Create `~/Library/LaunchAgents/com.user.truenas-backup.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.truenas-backup</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/username/bin/backup-to-truenas.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/username/truenas-backup.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/username/truenas-backup.err</string>
</dict>
</plist>
```

**Load the launchd job**:
```bash
launchctl load ~/Library/LaunchAgents/com.user.truenas-backup.plist
```

## Security Best Practices

### SSH Key Security

**1. Use strong key types and sizes**:
```bash
# RSA 4096-bit (good)
ssh-keygen -t rsa -b 4096

# Ed25519 (better, modern)
ssh-keygen -t ed25519 -C "your-email@example.com"
```

**2. Protect private keys with passphrases**:
- Adds encryption layer to private key
- Use `ssh-agent` to cache passphrase:
  ```bash
  # Add key to agent (enter passphrase once)
  ssh-add ~/.ssh/id_rsa

  # Verify key is loaded
  ssh-add -l

  # macOS: Store passphrase in Keychain
  ssh-add --apple-use-keychain ~/.ssh/id_rsa
  ```

**3. Set correct file permissions**:
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
chmod 600 ~/.ssh/config
chmod 644 ~/.ssh/known_hosts
```

**4. Use separate keys for different purposes**:
```bash
# Personal TrueNAS
ssh-keygen -t ed25519 -f ~/.ssh/truenas_personal -C "truenas-personal"

# Work TrueNAS
ssh-keygen -t ed25519 -f ~/.ssh/truenas_work -C "truenas-work"
```

Update `~/.ssh/config`:
```
Host truenas-personal
    HostName 192.168.1.100
    IdentityFile ~/.ssh/truenas_personal

Host truenas-work
    HostName 10.0.1.50
    IdentityFile ~/.ssh/truenas_work
```

### TrueNAS Security

**1. Use non-standard SSH port**:
- Change from default port 22 to reduce automated attacks
- Configure in TrueNAS: **Services** → **SSH** → **TCP Port**: `2222`
- Update `~/.ssh/config`: `Port 2222`

**2. Disable password authentication**:
- After verifying SSH keys work
- Configure in TrueNAS: **Services** → **SSH**
- **Allow Password Authentication**: Disabled
- Forces key-only authentication

**3. Restrict root login**:
- Configure in TrueNAS: **Services** → **SSH**
- **Log in as Root with Password**: Disabled
- Use regular users with `sudo` for admin tasks

**4. Use dedicated Rsync user**:
- Create user with minimal permissions
- Only grant access to backup directories
- Easier to audit and revoke access

**5. Configure firewall rules**:
- Restrict SSH access to specific IP addresses
- TrueNAS: **Network** → **Global Configuration** → **Firewall**
- Or use router/firewall rules

**6. Enable fail2ban** (if available):
- Blocks IPs after failed login attempts
- Check TrueNAS plugins/jails for fail2ban

**7. Monitor SSH logs**:
- Review regularly for suspicious activity
- TrueNAS: **System** → **Advanced** → **Syslog**

### Network Security

**1. Use VPN for remote access**:
- Don't expose TrueNAS SSH directly to internet
- Set up WireGuard or OpenVPN
- Access TrueNAS through VPN tunnel

**2. Local network only** (ideal):
- Only allow SSH connections from local network
- Use physical or site-to-site VPN for remote access

**3. Use static IP or hostname**:
- Easier to configure firewall rules
- More reliable for automated backups

## Troubleshooting

### SSH Key Authentication Fails

**Symptom**: Still prompted for password despite setting up SSH key.

**Solutions**:

1. **Verify public key on TrueNAS**:
   - Log into TrueNAS web interface
   - Check **Accounts** → **Users** → Your user → **SSH Public Key**
   - Ensure entire key is present (starts with `ssh-rsa` or `ssh-ed25519`)

2. **Check TrueNAS SSH home directory permissions**:

   SSH from Mac to TrueNAS (using password), then:
   ```bash
   # Check home directory permissions
   ls -la ~
   # Should be: drwx------ (700)

   # Check .ssh directory
   ls -la ~/.ssh
   # Should be: drwx------ (700)

   # Check authorized_keys
   ls -la ~/.ssh/authorized_keys
   # Should be: -rw------- (600)

   # Fix permissions if needed
   chmod 700 ~
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/authorized_keys
   ```

3. **Verify correct private key on Mac**:
   ```bash
   # Check key exists
   ls -la ~/.ssh/id_rsa

   # View public key fingerprint
   ssh-keygen -lf ~/.ssh/id_rsa.pub

   # Verify private key is valid
   ssh-keygen -y -f ~/.ssh/id_rsa
   ```

4. **Test with verbose output**:
   ```bash
   ssh -vvv username@truenas-ip
   ```

   Look for lines like:
   - `debug1: Offering public key: /Users/username/.ssh/id_rsa RSA SHA256:...`
   - `debug1: Server accepts key`
   - If you see `Permission denied (publickey)`, key authentication failed

5. **Check TrueNAS SSH logs**:
   - TrueNAS web interface: **System** → **Advanced** → **Syslog**
   - Or SSH to TrueNAS and check:
     ```bash
     tail -f /var/log/messages | grep sshd
     ```

### Connection Timeout

**Symptom**: Connection hangs or times out.

**Solutions**:

1. **Verify network connectivity**:
   ```bash
   ping 192.168.1.100
   ```

2. **Check SSH service is running**:
   - TrueNAS web interface: **Services** → **SSH** (should show green)

3. **Verify correct port**:
   ```bash
   # Test connection to SSH port
   nc -zv 192.168.1.100 22

   # Or with telnet
   telnet 192.168.1.100 22
   ```

4. **Check firewall rules**:
   - Ensure Mac can reach TrueNAS SSH port
   - Check router firewall
   - Check TrueNAS firewall settings

5. **Try different network**:
   - Test from different device on same network
   - Isolates whether issue is Mac-specific or network-wide

### Rsync Permission Denied

**Symptom**: Rsync fails with permission errors.

**Solutions**:

1. **Check destination directory permissions**:
   ```bash
   ssh truenas "ls -la /mnt/pool/backup"
   ```

2. **Verify user has write access**:
   - TrueNAS: **Storage** → **Pools** → Your pool
   - Navigate to destination directory
   - **Edit Permissions** → Add user with write access

3. **Test with simpler path**:
   ```bash
   # Try syncing to user's home directory first
   rsync -avz /tmp/test.txt truenas:~/
   ```

4. **Check dataset permissions**:
   - Some TrueNAS datasets have ACLs that override UNIX permissions
   - May need to adjust ACL entries

### Rsync Performance Issues

**Symptom**: Rsync is slow.

**Solutions**:

1. **Enable compression** (if not already):
   ```bash
   rsync -avzhP /local/path/ truenas:/remote/path/
   ```

2. **Use SSH compression**:
   ```bash
   rsync -avz -e "ssh -C" /local/path/ truenas:/remote/path/
   ```

3. **Disable compression for already-compressed files**:
   ```bash
   rsync -avhP --skip-compress=gz/zip/jpg/jpeg/png/mp4/mkv \
       /local/path/ truenas:/remote/path/
   ```

4. **Use rsync's delta-transfer algorithm**:
   ```bash
   # Already enabled by default, but can tune:
   rsync -avz --partial --inplace /local/path/ truenas:/remote/path/
   ```

5. **Check network bandwidth**:
   ```bash
   # Test raw SSH transfer speed
   ssh truenas "dd if=/dev/zero bs=1M count=100" | dd of=/dev/null
   ```

6. **Reduce SSH encryption overhead** (less secure):
   ```bash
   rsync -avz -e "ssh -c aes128-gcm@openssh.com" \
       /local/path/ truenas:/remote/path/
   ```

### Known Hosts Issues

**Symptom**: "Host key verification failed" or "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!"

**Solutions**:

1. **If TrueNAS was reinstalled or SSH keys regenerated**:
   ```bash
   # Remove old host key
   ssh-keygen -R 192.168.1.100

   # Or remove specific entry from known_hosts
   nano ~/.ssh/known_hosts
   ```

2. **Reconnect and accept new key**:
   ```bash
   ssh truenas
   # Type 'yes' when prompted
   ```

3. **Disable strict host checking** (less secure, not recommended):
   Add to `~/.ssh/config`:
   ```
   Host truenas
       StrictHostKeyChecking no
       UserKnownHostsFile /dev/null
   ```

## Summary

**Key points**:

1. **SSH keys enable secure, password-less authentication** between Mac and TrueNAS
2. **Generate keys on Mac**, add public key to TrueNAS user account
3. **Configure ~/.ssh/config** for simplified connections
4. **Rsync over SSH** provides efficient, secure file synchronization
5. **Security best practices**: Use strong keys, passphrases, non-default ports, disable password auth
6. **Automate backups** with cron or launchd for hands-off data protection

**Quick reference**:

```bash
# Generate key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Copy public key to TrueNAS (manual via web interface)
cat ~/.ssh/id_ed25519.pub

# Test connection
ssh truenas

# Sync to TrueNAS
rsync -avzhP /local/path/ truenas:/remote/path/

# Sync from TrueNAS
rsync -avzhP truenas:/remote/path/ /local/path/
```

## Related Topics

- [SSH Key Management Best Practices](#)
- [Automated Backup Strategies](#)
- [TrueNAS Security Hardening](#)
- [Network Storage Performance Optimization](#)
