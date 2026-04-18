# Publishing to GitHub - Step by Step Guide

This guide helps you publish your Diabetic Foot Ulcer and Retinopathy Detection project on GitHub.

## Prerequisites

- GitHub account at https://github.com/humanmnv224
- Git installed on your machine
- Project folder ready

## Steps to Publish

### 1. Create Repository on GitHub

1. Go to https://github.com/new
2. Enter repository name: `Diabetic_Foot_Ulcer_And_Retinopathy_Detection`
3. Description: `Multimodal AI screening system combining DFU severity grading and diabetic retinopathy detection`
4. Choose **Public** (so anyone can view)
5. Check "Add a README file" - **NO** (we already have one)
6. Check "Add .gitignore" - **NO** (we already have it)
7. Choose License: **MIT License**
8. Click "Create repository"

### 2. Initialize Git Locally (if not already done)

```powershell
cd C:\Users\valva\OneDrive\Desktop\Diabetic_Foot_Ulcer_And_Retinopathy_Detection
git init
git add .
git commit -m "Initial commit: DFU and Retinopathy detection project"
```

### 3. Connect to GitHub and Push

Replace `USERNAME` with your GitHub username in the URL:

```powershell
git remote add origin https://github.com/humanmnv224/Diabetic_Foot_Ulcer_And_Retinopathy_Detection.git
git branch -M main
git push -u origin main
```

If prompted for credentials:
- Use GitHub personal access token (recommended over password)
- Generate at: https://github.com/settings/tokens/new
- Scopes needed: `repo`, `read:user`, `user:email`

### 4. Verify on GitHub

1. Go to https://github.com/humanmnv224/Diabetic_Foot_Ulcer_And_Retinopathy_Detection
2. Check that all files appear
3. README.md displays correctly
4. Verify `.venv/` and sensitive files are NOT visible (excluded by .gitignore)

## Files Included (Public)

✅ Source code:
- `dfu_progression_project/src/`
- `diabetic-retinopathy-detection-main/src/`
- `*.py` files (model, training, inference)

✅ Configuration:
- `dfu_progression_project/conf/config.yaml`
- `diabetic-retinopathy-detection-main/conf/config.yaml`

✅ Documentation:
- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `.gitignore`

✅ Pre-trained models (if included):
- `dfu_progression_project/artifacts/checkpoints/`
- `diabetic-retinopathy-detection-main/artifacts/dr-model.ckpt`

## Files Excluded (Private - Not Published)

❌ Virtual environment:
- `.venv/`
- `.venv-1/`

❌ Sensitive data:
- `.env` (environment variables)
- `*.env.local`

❌ Generated/temporary:
- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `logs/`

❌ Large datasets (optional):
- `DFU_Dataset/` (if > 100MB, consider using Git LFS)
- `dfu_progression_project/dataset/` (training data)

## Future Updates

After initial push, keep your repo updated:

```powershell
git add .
git commit -m "Describe your changes"
git push origin main
```

## Setting Up GitHub Pages (Optional)

To host documentation:

1. Go to Settings → Pages
2. Select "Deploy from a branch"
3. Choose `main` branch
4. Save

Your README becomes accessible at:
`https://humanmnv224.github.io/Diabetic_Foot_Ulcer_And_Retinopathy_Detection/`

## Badges for README (Optional)

Add to your README top section:

```markdown
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![License](https://img.shields.io/badge/License-MIT-green)
```

---

You're ready to publish! 🚀

Questions? See CONTRIBUTING.md or GitHub documentation.
