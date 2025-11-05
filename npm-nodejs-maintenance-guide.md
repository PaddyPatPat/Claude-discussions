# NPM and Node.js Maintenance Guide

A comprehensive guide for maintaining, updating, and troubleshooting your npm and Node.js installation, with a focus on nvm (Node Version Manager) workflows.

## Initial Health Check

Before making any changes, assess your current setup:

```bash
# Check versions
npm --version
node --version

# List global packages
npm list -g --depth=0

# Run comprehensive diagnostics
npm doctor
```

### Understanding npm doctor Output

The `npm doctor` command checks:
- **npm version**: Whether you're using the latest
- **Node.js version**: Whether you're using the recommended version
- **Registry connectivity**: Can reach npmjs.org
- **Git executable**: Required for installing from Git repos
- **Global bin folder**: Proper PATH configuration
- **Permissions**: On cached files, node_modules, and bin folders
- **Cache integrity**: Verifies cached packages

## Updating NPM

### When Using nvm (Recommended)

If you're using nvm (Node Version Manager), update Node.js first - it comes with npm:

```bash
# Install latest Node.js with latest npm
nvm install node --latest-npm

# Use the new version
nvm use node

# Set as default for new shells
nvm alias default node
```

### When Using System Node.js

If you installed Node.js directly (not through nvm):

```bash
# Update npm globally
npm install -g npm@latest

# Verify the update
npm --version
```

**Important**: Always use `-g` (global) flag, not `-f` (force) when updating npm.

## Cache Management

NPM caches downloaded packages to speed up installations. Over time, this cache can become corrupted or outdated.

### Clean Cache

```bash
# Clear npm cache (use with caution)
npm cache clean --force

# Verify cache integrity
npm cache verify
```

### When to Clean Cache

- After major npm or Node.js updates
- When experiencing strange installation errors
- When packages fail to install correctly
- When `npm doctor` reports cache issues

**Note**: The `--force` flag is required for `npm cache clean` because npm wants you to be sure - clearing cache means re-downloading packages.

## Using NVM (Node Version Manager)

### Why NVM?

NVM allows you to:
- Run multiple Node.js versions side-by-side
- Switch between versions per project
- Avoid permission issues with global packages
- Test code across different Node.js versions

### How NVM Isolates Versions

Each Node.js version installed through nvm is completely separate:

```
~/.nvm/versions/node/
├── v22.8.0/
│   ├── bin/
│   │   ├── node
│   │   └── npm
│   └── lib/
│       └── node_modules/  # Global packages for v22
│           ├── typescript
│           └── eslint
└── v24.6.0/
    ├── bin/
    └── lib/
        └── node_modules/  # Separate globals for v24
            └── (empty initially)
```

**Key Insight**: When you switch Node.js versions with nvm, you switch to a completely different set of global packages.

### Version Switching Implications

When you run `nvm use v24.6.0` after having used `v22.8.0`:

**What Happens**:
- Your PATH changes to point to v24.6.0's binaries
- Global packages from v22.8.0 are no longer accessible
- You need to reinstall global packages for v24.6.0

**What Doesn't Happen**:
- Your old global packages aren't deleted (still in v22.8.0's directory)
- Your project-local `node_modules` are unaffected
- Your npm configuration carries over

### Managing Global Packages Across Versions

After switching to a new Node.js version:

```bash
# Check what's currently available
npm list -g --depth=0

# Likely you'll see only:
# ├── npm@version
# └── corepack@version

# Reinstall your essential global tools
npm install -g typescript eslint prettier
```

### Version Management Commands

```bash
# List installed Node.js versions
nvm list

# List available Node.js versions
nvm list-remote

# Install specific version
nvm install v22.18.0

# Install with latest npm
nvm install v22.18.0 --latest-npm

# Use specific version
nvm use v22.18.0

# Set default version for new shells
nvm alias default v22.18.0

# Use version specified in .nvmrc file
nvm use
```

### Project-Specific Node.js Versions

Create a `.nvmrc` file in your project root:

```
22.18.0
```

Then:
```bash
# In your project directory
nvm use
# Automatically uses the version from .nvmrc
```

## Update Strategy

### Recommended Update Order

1. **Update Node.js** (brings npm with it)
   ```bash
   nvm install node --latest-npm
   nvm use node
   nvm alias default node
   ```

2. **Update npm** (if needed)
   ```bash
   npm install -g npm@latest
   ```

3. **Verify everything**
   ```bash
   node --version
   npm --version
   npm doctor
   ```

4. **Reinstall global packages**
   ```bash
   npm install -g typescript eslint prettier [other tools]
   ```

5. **Clean cache**
   ```bash
   npm cache clean --force
   npm cache verify
   ```

### What Not To Do

```bash
# ❌ Don't use -f (force) flag to install npm
npm install -f npm@latest

# ✅ Use -g (global) flag instead
npm install -g npm@latest
```

The `-f` flag forces installation even if there are conflicts, which can cause issues. Use `-g` for global installations.

## Managing Global Packages

### List Global Packages

```bash
# Show only top-level global packages
npm list -g --depth=0

# Show with location
npm list -g --depth=0 --long
```

### Update Global Packages

```bash
# Check which globals are outdated
npm outdated -g --depth=0

# Update all global packages (use with caution)
npm update -g

# Update specific package
npm update -g typescript
```

### Typical Global Packages for Development

Common tools installed globally:

```bash
npm install -g typescript        # TypeScript compiler
npm install -g eslint           # JavaScript linter
npm install -g prettier         # Code formatter
npm install -g @typescript-eslint/eslint-plugin  # TypeScript linting
npm install -g @typescript-eslint/parser         # TypeScript parser for ESLint
```

## Troubleshooting

### npm doctor Failures

**Issue: "npm version not latest"**
```bash
npm install -g npm@latest
```

**Issue: "node version not recommended"**
```bash
# With nvm
nvm install node --latest-npm
nvm use node

# Without nvm
# Download and install from nodejs.org
```

**Issue: "Cache issues"**
```bash
npm cache clean --force
npm cache verify
```

**Issue: "Permission errors"**

If not using nvm:
```bash
# Create directory for global packages in home directory
mkdir ~/.npm-global

# Configure npm to use new directory
npm config set prefix '~/.npm-global'

# Add to PATH in ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export PATH=~/.npm-global/bin:$PATH
```

**Best solution**: Use nvm to avoid permission issues entirely.

### Common Errors and Solutions

**Error: EACCES permission denied**
- **Cause**: Trying to install globally without proper permissions
- **Solution**: Use nvm, or fix npm permissions as shown above

**Error: Maximum call stack size exceeded**
- **Cause**: Corrupted npm cache or circular dependencies
- **Solution**: `npm cache clean --force` then retry

**Error: Cannot find module**
- **Cause**: Incomplete installation or cache issues
- **Solution**: Delete `node_modules` and `package-lock.json`, then `npm install`

**Warning: deprecated packages**
- **Cause**: Package dependencies use old packages
- **Solution**: Usually safe to ignore unless security-related; update project dependencies

## Best Practices

### For Development Environment

1. **Use nvm**: Provides version isolation and avoids permission issues
2. **Keep Node.js and npm updated**: Security and performance improvements
3. **Run npm doctor periodically**: Catch issues early
4. **Clean cache after major updates**: Prevents stale package issues
5. **Document global packages**: Keep a list of essential globals for easy reinstallation

### Global vs Project Packages

**Install globally**:
- CLI tools you use across projects (typescript, eslint)
- Development utilities (prettier, nodemon)
- Package managers (npm, yarn)

**Install per-project**:
- Project dependencies (express, react, lodash)
- Testing frameworks (jest, mocha)
- Build tools (webpack, vite)

See [development-tool-installation-strategies.md](development-tool-installation-strategies.md) for more details.

### Regular Maintenance Schedule

**Monthly**:
- Check for npm updates: `npm outdated -g --depth=0`
- Update critical packages

**Quarterly**:
- Update Node.js to latest LTS
- Run `npm doctor`
- Clean cache: `npm cache clean --force`

**Before major project**:
- Verify setup: `npm doctor`
- Update all tools: `npm update -g`

## Version Strategy

### LTS vs Current

- **LTS (Long Term Support)**: Stable, recommended for production
  - Even-numbered versions (22, 24, 26)
  - Security updates for 30 months
  - Best for most developers

- **Current**: Latest features, faster updates
  - Odd-numbered versions (23, 25, 27)
  - Becomes LTS or deprecated after 6 months
  - For testing new features

### Recommended Approach

```bash
# For most development work
nvm install --lts

# For bleeding edge
nvm install node

# Check what's active
nvm current

# See all installed versions
nvm list
```

## Migration Checklist

When updating to a new Node.js version:

- [ ] Note current Node.js version: `node --version`
- [ ] List current global packages: `npm list -g --depth=0 > globals.txt`
- [ ] Install new Node.js version: `nvm install node --latest-npm`
- [ ] Switch to new version: `nvm use node`
- [ ] Set as default: `nvm alias default node`
- [ ] Verify versions: `node --version && npm --version`
- [ ] Run diagnostics: `npm doctor`
- [ ] Reinstall global packages from globals.txt
- [ ] Test project: `npm install && npm test`
- [ ] Clean cache: `npm cache clean --force`

## Quick Reference

```bash
# Health Check
npm doctor
npm list -g --depth=0
npm outdated -g --depth=0

# Updates (with nvm)
nvm install node --latest-npm
nvm use node
nvm alias default node
npm install -g npm@latest

# Updates (without nvm)
npm install -g npm@latest

# Maintenance
npm cache clean --force
npm cache verify
npm update -g

# Version Management (nvm)
nvm list
nvm install <version>
nvm use <version>
nvm alias default <version>
```

## Summary

- **Use nvm** for managing Node.js versions and avoiding permission issues
- **Update Node.js first**, which brings npm with it
- **Understand version isolation**: Each nvm Node.js version has separate global packages
- **Run npm doctor** regularly to catch issues early
- **Clean cache** after major updates to prevent stale package problems
- **Keep a list** of your essential global packages for easy reinstallation after version updates

For project-specific package management strategies, see [development-tool-installation-strategies.md](development-tool-installation-strategies.md).
