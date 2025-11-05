# Claude Code Installation Methods

Claude Code can be installed using two different methods. This guide compares them and provides a recommendation.

## Installation Options

### 1. npm Installation (Traditional Method)

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Native Binary Installation (Recommended)

```bash
# macOS/Linux
curl -fsSL https://claude.ai/install.sh | bash

# Windows
irm https://claude.ai/install.ps1 | iex
```

## Key Differences

### npm Installation

**Pros**:
- Familiar installation method for Node.js developers
- Easy to integrate into existing Node.js workflows
- Cross-platform (works wherever Node.js works)
- Version management through npm

**Cons**:
- Requires Node.js 18+ as a dependency
- Can lead to permission issues (DO NOT use `sudo npm install -g`)
- More complex setup for CI/CD environments
- Slower startup (Node.js overhead)

### Native Binary Installation (Recommended)

**Pros**:
- No Node.js dependency required
- Automatic updates - downloads and installs automatically in the background
- Better performance (native binary)
- Removes outdated aliases or symlinks automatically
- Simpler permission model

**Cons**:
- Newer method (may have rough edges)
- Native Windows support was just added recently
- Less familiar to Node.js developers

## Migration Path

If you have an existing npm installation of Claude Code:
1. Use `claude install` to start the native binary installation
2. After global install via npm, use `claude migrate-installer` to move to native installation

## Recommendation

**Use the native binary installation** for most use cases because:
1. **Simpler setup**: No Node.js version management concerns
2. **Better performance**: Native binary runs faster
3. **Automatic updates**: Claude Code automatically keeps itself up to date to ensure you have the latest features and security fixes
4. **Future-proof**: This seems to be Anthropic's preferred distribution method going forward

## Verification

After installing, verify your installation:

```bash
# Check installation
claude doctor

# Verify version
claude --version

# Check location
which claude
```

### Expected Output (Native Installation)

```bash
$ which claude
/Users/username/.local/bin/claude  # Not in npm directories

$ claude doctor
✅ Claude Code installation verified
✅ Currently running: native (1.0.90)
✅ Auto-updates enabled: true
✅ Search: OK (bundled)
```

### Expected Output (npm Installation)

```bash
$ which claude
/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude

$ claude --version
1.0.90 (Claude Code)
# Note: May not have auto-update feature
```

## Installation for Specific Use Cases

### For Learning and Emacs Integration
**Recommendation**: Native binary installation
- Cleaner installation
- Optimized for current Claude Code architecture
- Works seamlessly with Emacs integration

### For Node.js-Heavy Development Environments
**Consider**: npm installation if you already have strict Node.js version requirements
- Better integration with existing Node.js tooling
- Easier to include in `package.json` for team projects

### For CI/CD Pipelines
**Recommendation**: Native binary installation
- Simpler CI/CD setup
- No Node.js version conflicts
- Faster execution

## Removing npm Installation

If you want to switch from npm to native:

```bash
# Remove the global npm package
npm uninstall -g @anthropic-ai/claude-code

# Verify removal
which claude  # Should show "command not found"

# Clear npm cache
npm cache clean --force

# Install native binary
curl -fsSL https://claude.ai/install.sh | bash
```

## Official Documentation

For the most up-to-date installation instructions, visit:
https://docs.anthropic.com/en/docs/claude-code/quickstart
