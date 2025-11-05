# Development Tool Installation Strategies

Understanding the difference between global and project-specific installation of development tools, particularly for Node.js/npm packages.

## Global vs Project-Specific Installation

### Global Installation

```bash
npm install -g eslint prettier
```

**What it does**:
- Installs tools system-wide (available everywhere)
- Tools go into a global directory (like `/usr/local/lib/node_modules/`)
- Same version used across all projects

**Use cases**:
- Command-line tools you use frequently across projects
- System utilities
- Tools you want available in any directory

### Project-Specific Installation (Recommended)

```bash
npm install --save-dev eslint prettier
```

**What it does**:
- Installs tools per project (only for current project)
- Tools go into `./node_modules/.bin/` in your project directory
- Each project can have different versions
- The `--save-dev` flag adds them to `package.json` as development dependencies

**Use cases**:
- Linters specific to your project
- Build tools for your application
- Testing frameworks
- Any tool that different projects might need different versions of

## Why Project-Specific is Recommended

### 1. Version Consistency Across Team/Time

```json
// package.json after running --save-dev
{
  "devDependencies": {
    "eslint": "^8.45.0",
    "prettier": "^3.0.0"
  }
}
```

**Benefits**:
- Team consistency: Everyone uses the same linter versions
- Reproducible builds: Project works the same way months/years later
- No "works on my machine" issues

### 2. Different Projects, Different Needs

```
project-a/
├── package.json  (eslint v8, stricter rules)
└── node_modules/

project-b/
├── package.json  (eslint v7, legacy codebase)
└── node_modules/
```

**Why this matters**:
- Legacy projects might need older linter versions
- Different coding standards per project
- Framework-specific rules (React vs Vue vs vanilla JS)

### 3. Avoiding Global Pollution

**Problems with global installation**:
- Permission issues (might need sudo on some systems)
- Version conflicts between projects
- Hard to track what tools are installed globally
- Difficult onboarding for new team members

## How Tools Find Executables

### Project-Specific Detection (Preferred)

Flycheck and other tools automatically look for executables in this order:
1. `./node_modules/.bin/eslint` (project-specific)
2. `/usr/local/bin/eslint` (global)
3. `eslint` in `$PATH` (system)

### What This Means Practically

```
your-project/
├── .eslintrc.js          # Project-specific rules
├── package.json          # Specifies eslint version
├── node_modules/
│   └── .bin/
│       ├── eslint        # Flycheck will use THIS one
│       └── prettier      # Not the global version
└── src/
    └── main.js          # Gets linted with project rules
```

## Real-World Example

### Scenario: Working on Multiple JavaScript Projects

**Project A: Modern React App**
```bash
cd project-a
npm install --save-dev eslint@8.45.0 prettier@3.0.0 @typescript-eslint/parser
```

- Uses latest ESLint with TypeScript support
- Strict modern JavaScript rules
- Prettier with 2-space indentation

**Project B: Legacy jQuery Site**
```bash
cd project-b
npm install --save-dev eslint@7.32.0 prettier@2.8.0
```

- Uses older ESLint compatible with ES5
- More permissive rules for older code
- Prettier with 4-space indentation (team preference)

**Result**:
- Flycheck automatically uses the right version per project
- No conflicts or "this rule doesn't exist" errors
- Each project maintains its own standards

## Setup Steps for Project-Specific Installation

### 1. Initialize Your Project (if not already done)

```bash
cd your-project
npm init -y  # Creates package.json
```

### 2. Install Development Tools

```bash
# Install linters as dev dependencies
npm install --save-dev eslint prettier

# Install project-specific ESLint config
npm install --save-dev @eslint/js  # Basic JavaScript rules
```

### 3. Create Project Configuration

```bash
# Generate ESLint config
npx eslint --init

# Create prettier config
echo '{"semi": true, "singleQuote": true}' > .prettierrc
```

### 4. Verify Tool Detection

Open a JavaScript file in Emacs and run:
```
M-x flycheck-verify-setup
```

You should see something like:
```
- may enable:  Flycheck checker javascript-eslint
  - executable:  Found at /your-project/node_modules/.bin/eslint
  - configuration file: Found at /your-project/.eslintrc.js
```

## Package.json Scripts

You can also add scripts to run these tools:

```json
{
  "scripts": {
    "lint": "eslint src/",
    "format": "prettier --write src/**/*.js",
    "lint:fix": "eslint src/ --fix"
  }
}
```

Then run with:
```bash
npm run lint
npm run format
```

## When to Use Global Installation

Despite the recommendation for project-specific installation, global installation is appropriate for:

- **Command-line utilities** you use across all projects
- **One-off tools** that don't affect project builds
- **System-level tools** like `npm` itself
- **Personal productivity tools** that aren't part of the project workflow

Examples:
```bash
npm install -g http-server    # Quick local server
npm install -g json-server    # Mock API server
npm install -g tldr           # Command help
```

## Python Equivalent

The same principles apply to Python:

**Global** (similar issues):
```bash
pip install flake8 pylint
```

**Project-specific** (recommended):
```bash
# Using virtual environments
python -m venv venv
source venv/bin/activate
pip install flake8 pylint

# Or using poetry/pipenv for project isolation
```

## Best Practices

1. **Default to project-specific** unless you have a specific reason for global
2. **Document in package.json** so teammates get the same versions
3. **Check in package.json and package-lock.json** to version control
4. **Don't check in node_modules/** to version control
5. **Use .gitignore** to exclude `node_modules/`

## Troubleshooting

### Tool not found by Flycheck

Check:
1. Is it in `package.json`?
2. Did you run `npm install`?
3. Is `node_modules/.bin/` executable?
4. Run `M-x flycheck-verify-setup` to see what Flycheck detects

### Version conflicts

If you have both global and local installations:
- Flycheck should prefer local (project-specific)
- Verify with `which eslint` vs `./node_modules/.bin/eslint`
- Consider uninstalling global versions to avoid confusion

## Summary

**Project-specific installation is recommended** because it ensures version consistency, prevents conflicts between projects, and makes onboarding new team members easier. While global installation is simpler initially, the benefits of project-specific installation far outweigh the small additional setup cost.
