# After Committing Changes in a Feature Branch

## Push Your Feature Branch

1. Push your committed changes to your fork:
```bash
git push origin init_steven
```

## Update Main (Optional but Recommended)

1. Switch to main branch:
```bash
git checkout main
```

2. Get latest changes from upstream:
```bash
git fetch upstream
git merge upstream/main
```

3. Push updated main to your fork:
```bash
git push origin main
```

## Next Steps

1. Go to GitHub and navigate to your fork
2. You should see a notification to create a Pull Request for your recently pushed branch
3. Click "Compare & pull request" to start the PR process

> Note: Do not push directly to main. Always work through feature branches and pull requests.
