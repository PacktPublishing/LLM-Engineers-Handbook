# Working with a Forked Repository in VS Code

## Initial Setup

1. Fork the repository on GitHub by clicking the "Fork" button at the top right of the repository page.

2. Clone your forked repository locally:
```bash
git clone https://github.com/stevenmichiels/LLM-Engineers-Handbook.git
cd LLM-Engineers-Handbook
```

3. Add the original repository as a remote called "upstream":
```bash
git remote add upstream https://github.com/PacktPublishing/LLM-Engineers-Handbook.git
```

> Note: Make sure you fork from the original repository at https://github.com/PacktPublishing/LLM-Engineers-Handbook to stay up-to-date with the main project.

## Creating and Working in Your Branch

1. Create and switch to a new branch:
```bash
git checkout -b your-branch-name
```

2. Make your changes in VS Code.

## Keeping Your Fork Updated

### Understanding the Setup

Your local repository has two remote connections:
- `origin`: Your fork on GitHub (your copy)
- `upstream`: The original repository you forked from

```
[Original Repo (upstream)] ← → [Your Fork (origin)] ← → [Your Local Repo]
```

### Step-by-Step Sync Process

1. **Fetch upstream changes**
   ```bash
   git fetch upstream
   ```
   This downloads all new changes from the original repository but doesn't apply them yet.
   
2. **Switch to your main branch**
   ```bash
   git checkout main
   ```
   Ensures you're updating your main branch first.
   
3. **Merge upstream changes**
   ```bash
   git merge upstream/main
   ```
   This applies the changes from the original repo to your local main branch.
   
4. **Update your fork on GitHub**
   ```bash
   git push origin main
   ```
   Pushes the updated main branch to your fork on GitHub.

### Updating Your Feature Branch

After your main branch is updated, you'll want to get those changes into your feature branch:

1. **Switch to your feature branch**
   ```bash
   git checkout your-branch-name
   ```

2. **Merge from main**
   ```bash
   git merge main
   ```
   This brings all the updates into your working branch.

### Common Scenarios

1. **When to sync:**
   - Before starting new work
   - When the original repository has new changes
   - Before creating a pull request

2. **Handling merge conflicts:**
   If you get merge conflicts:
   - VS Code will highlight conflicts in the editor
   - Resolve each conflict by choosing which changes to keep
   - After resolving, use `git add .` to mark as resolved
   - Complete the merge with `git commit`

3. **Quick sync command sequence:**
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   git checkout your-branch-name
   git merge main
   ```

### Visual Example

```
Before sync:
upstream/main:    A → B → C
                  │   │   │
                  │   │   └── (Latest commit in upstream: e.g., new feature added)
                  │   └────── (Second commit: e.g., documentation update)
                  └────────── (Initial commit that all branches share)

your/main:        A → B
                  └── (Your fork is 1 commit behind upstream)

your/feature:     A → B → X
                          │
                          └── (Your new changes in your feature branch)

After sync:
upstream/main:    A → B → C
your/main:        A → B → C
your/feature:     A → B → X → C
                          │   │
                          │   └── (Upstream changes merged into your feature)
                          └────── (Your changes preserved)
```

Letters represent commits:
- `A`: Initial commit that exists in all branches
- `B`: Later commit that your fork already has
- `C`: New commit in the upstream repo that you need to sync
- `X`: Your new changes in your feature branch

## VS Code Specific Tips

- Use the Source Control panel (Ctrl+Shift+G) to manage your changes
- The bottom status bar shows your current branch
- Use the VS Code integrated terminal for git commands
- Install the "GitHub Pull Requests and Issues" extension for better GitHub integration

## Best Practices

- Always pull from upstream before starting new work
- Create a new branch for each feature or fix
- Keep commits atomic and well-described
- Regularly push your branch to your fork
