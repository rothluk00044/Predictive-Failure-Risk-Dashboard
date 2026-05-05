# Windows Setup Guide for Predictive Failure Risk Dashboard

If you see: `Python was not found; run without arguments to install from the Microsoft Store…`, follow this guide to fix it.

---

## Step 1: Install Python

### Option A: Download from python.org (Recommended)

1. Open [python.org/downloads](https://www.python.org/downloads/)
2. Click **"Download Python 3.11"** (or the latest 3.x version)
3. Run the installer (`.exe` file)
4. **⚠️ CRITICAL**: In the installer window, check the box:
   - ✅ **"Add Python to PATH"** (bottom left of the first screen)
5. Click **"Install Now"** (or "Customize Installation" if you need specific options)
6. Wait for installation to complete (2-3 minutes)
7. **Close the installer**

### Option B: Windows Store (Alternative)

1. Open **Microsoft Store** (Windows key → type "Store")
2. Search for **"Python 3.11"** (or latest)
3. Click **"Install"**
4. Wait for installation
5. **Open a new PowerShell/Command Prompt window** (critical for PATH to update)

---

## Step 2: Verify Python Installation

Open a **new PowerShell window** (this ensures environment updates take effect):

1. Click **Windows key** → Type **"PowerShell"** → Right-click → **"Run as Administrator"** (optional but safe)
2. Run these commands:

```powershell
python --version
```

**Expected output:**
```
Python 3.11.x (or whatever version you installed)
```

If you still see `Python was not found`, try:
```powershell
py --version
```

If `py --version` works but `python --version` doesn't, Python is installed but not fully on PATH. **Restart your computer** to apply PATH changes globally.

### Verify pip (package manager)

```powershell
python -m pip --version
```

**Expected output:**
```
pip 23.x.x from C:\Users\...\Python311\lib\site-packages\pip (python 3.11)
```

---

## Step 3: Navigate to Project Directory

```powershell
cd C:\Users\rothl\PFRD\Predictive-Failure-Risk-Dashboard
```

Verify you're in the right place:
```powershell
ls
```

You should see: `app.py`, `train_model.py`, `requirements.txt`, `README.md`, etc.

---

## Step 4: Install Project Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**What this does**: Installs pandas, scikit-learn, streamlit, joblib, matplotlib, numpy, ucimlrepo.

**Expected final output** (last few lines):
```
Successfully installed [packages...]
```

If you see warnings like `WARNING: Retrying...`, it's usually fine. If you see `ERROR`, check your internet connection and retry.

---

## Step 5: Train the Model

```powershell
python train_model.py
```

**What happens**:
- Downloads UCI AI4I dataset (~2 MB, takes 10-30 seconds depending on connection)
- Trains RandomForest model (~30 seconds to 2 minutes)
- Prints confusion matrix and classification report to console
- Saves `failure_model.pkl` and `model_features.pkl`

**Expected output** (last lines):
```
[6] Saving Model...
    Saved: failure_model.pkl, model_features.pkl

============================================================
Training complete. Ready to run dashboard with:
  python -m streamlit run app.py
============================================================
```

✅ **If you see this, training succeeded!**

---

## Step 6: Run the Dashboard

```powershell
python -m streamlit run app.py
```

**What happens**:
1. Streamlit starts a local web server
2. Browser automatically opens to `http://localhost:8501`
3. You'll see the Predictive Failure Risk Dashboard

**Expected console output** (first 10 lines):
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  For better performance, install pyarrow: pip install pyarrow
```

✅ **Dashboard is running!**

### Using the Dashboard

- **Sidebar sliders**: Adjust operating conditions (temperature, speed, torque, wear)
- **Risk panel**: See real-time failure risk predictions (%)
- **Risk category**: Low (green), Medium (orange), or High (red)
- **Recommendations**: Engineering advice based on risk level
- **Feature importance**: Chart showing which factors matter most

**To stop the dashboard**: In PowerShell, press `Ctrl + C`

---

## Troubleshooting

### Problem: `Python was not found`

**Solutions** (in order):
1. Restart PowerShell/Command Prompt (sometimes PATH doesn't update immediately)
2. Restart your computer
3. Reinstall Python, making sure to check "Add Python to PATH"
4. Check Environment Variables manually:
   - Windows key → type "environment" → click "Edit environment variables for your account"
   - Look for `PATH` in "User variables"
   - It should contain something like `C:\Users\rothl\AppData\Local\Programs\Python\Python311`
   - If missing, add it manually, then restart PowerShell

### Problem: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```powershell
python -m pip install streamlit
```

Or reinstall all dependencies:
```powershell
python -m pip install -r requirements.txt
```

### Problem: `FileNotFoundError: failure_model.pkl`

**Solution**: Train the model:
```powershell
python train_model.py
```

### Problem: `ucimlrepo` download fails (network error)

**Solution**: The UCI dataset service may be temporarily unavailable. Wait a few minutes and retry:
```powershell
python train_model.py
```

If it keeps failing, check your internet connection.

### Problem: Streamlit opens but buttons/sliders don't work

**Solution**: Refresh the browser (F5 or Ctrl + R)

### Problem: Port 8501 already in use

**Solution**: Either:
- Close any other Streamlit app running on your machine
- Or use a different port:
  ```powershell
  python -m streamlit run app.py --server.port=8502
  ```

---

## Quick Reference

```powershell
# Once per machine (install Python + pip)
# [Follow Steps 1–2 above]

# Once per project (install dependencies)
cd C:\Users\rothl\PFRD\Predictive-Failure-Risk-Dashboard
python -m pip install -r requirements.txt

# Once per training session (train model)
python train_model.py

# Every time you want to use the dashboard
python -m streamlit run app.py

# Stop the dashboard
Ctrl + C
```

---

## Next Steps After Setup

1. ✅ Verify the dashboard works by adjusting sliders and watching predictions update
2. 📖 Read [README.md](README.md) for project context and architecture
3. 🔍 Explore the code:
   - `train_model.py`: How the model is trained
   - `app.py`: How the dashboard is built
4. 💡 Try modifications:
   - Adjust RandomForest hyperparameters in `train_model.py`
   - Add new input fields in `app.py`
   - Deploy to the cloud (Streamlit Cloud is free!)

---

## Getting Help

- **Python not found?** → Restart computer, check PATH
- **Package installation fails?** → Check internet, try `pip install --upgrade pip` first
- **Streamlit crashes?** → Check PyCharm/VS Code terminal for error messages, restart
- **Model training is slow?** → Normal (first run downloads ~2 MB dataset); subsequent runs are faster

---

## Success Checklist

- [ ] `python --version` shows Python 3.8+
- [ ] `python -m pip --version` works without errors
- [ ] `python train_model.py` completes with "Training complete"
- [ ] `failure_model.pkl` exists in the project folder
- [ ] `python -m streamlit run app.py` opens a browser window
- [ ] Sliders in the dashboard respond to input
- [ ] Risk prediction updates in real-time

✅ **If all boxes are checked, you're ready to demo this in your Gates interview!**
