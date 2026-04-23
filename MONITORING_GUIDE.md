# 📊 Monitoring Fine-Tuning Metrics

## ✅ What's Now Enabled

Your training script now tracks these metrics (just like in the image):

1. **loss** - Training loss (should decrease)
2. **eval_loss** - Validation loss (should decrease)
3. **learning_rate** - Learning rate schedule (linear decay)
4. **grad_norm** - Gradient norm (indicates training stability)

---

## 🎯 Two Ways to Monitor

### Option 1: TensorBoard (Local, FREE) ⭐ ENABLED BY DEFAULT

**During training**, open a new terminal and run:
```bash
tensorboard --logdir output_mac/runs
```

Then open in browser: **http://localhost:6006**

**You'll see**:
- Real-time graphs updating as training progresses
- All 4 metrics: loss, eval_loss, learning_rate, grad_norm
- Exactly like the image shown, but in TensorBoard UI

**Pros**:
- ✅ Free, runs locally
- ✅ No account needed
- ✅ Real-time updates
- ✅ Works offline

---

### Option 2: Comet ML (Cloud, Professional) 🌐 OPTIONAL

**To enable** (like in the image):

1. Get Comet ML API key: https://www.comet.com/signup
2. Set environment variable:
   ```bash
   export COMET_API_KEY="your_api_key"
   ```
3. Install Comet ML:
   ```bash
   pip install comet-ml
   ```
4. Edit `tools/finetune_mac.py`:
   ```python
   use_comet_ml = True  # Change from False
   ```

**You'll get**:
- ✅ Professional dashboard (like the image)
- ✅ Cloud storage of experiments
- ✅ Compare multiple runs
- ✅ Team collaboration

---

## 📈 What Each Metric Means

### 1. Training Loss
- **What**: How well model fits training data
- **Good**: Steadily decreasing from ~1.2 to ~0.5-0.7
- **Bad**: Not decreasing, or jumping erratically

### 2. Evaluation Loss (eval_loss)
- **What**: How well model performs on unseen test data
- **Good**: Decreasing, similar to training loss
- **Bad**: Increasing (overfitting), or much higher than training loss

### 3. Learning Rate
- **What**: How big the parameter updates are
- **Good**: Smooth linear decrease from 3e-4 to 0
- **Pattern**: Starts high, gradually reduces

### 4. Gradient Norm (grad_norm)
- **What**: Size of gradients during backprop
- **Good**: Stable around 0.1-0.2
- **Bad**: Very high spikes (>1.0) or zeros

---

## 🖥️ Example: Viewing with TensorBoard

**Step 1**: Start training
```bash
export HF_TOKEN="your_token"
python tools/finetune_mac.py
```

**Step 2**: In a NEW terminal (while training runs)
```bash
cd /Users/sumantopal/personal/LLM-Engineers-Handbook
tensorboard --logdir output_mac/runs
```

**Step 3**: Open browser
```
http://localhost:6006
```

**You'll see**:
```
SCALARS tab:
  📉 loss (orange line going down)
  📉 eval_loss (blue line going down)
  📉 learning_rate (smooth decay)
  📊 grad_norm (stable line)
```

---

## 📊 Expected Training Metrics

Based on similar training runs:

| Metric | Start | Mid-Training | End |
|--------|-------|--------------|-----|
| **loss** | ~1.2 | ~0.8 | ~0.5-0.7 |
| **eval_loss** | ~0.82 | ~0.77 | ~0.75-0.76 |
| **learning_rate** | 3e-4 | 1.5e-4 | ~0 |
| **grad_norm** | ~0.6 | ~0.15-0.2 | ~0.15-0.2 |

---

## 🚨 Warning Signs to Watch

### ❌ Bad Signs:
- Loss not decreasing after 100 steps
- Eval loss increasing (overfitting)
- Grad norm > 1.0 consistently (gradient exploding)
- Loss becomes NaN (training crashed)

### ✅ Good Signs:
- Smooth loss decrease
- Eval loss tracking training loss
- Stable grad norm
- Learning rate smoothly decreasing

---

## 💡 Pro Tips

1. **Check TensorBoard every 30 minutes** to catch issues early
2. **Compare to expected values** in the table above
3. **If loss isn't decreasing**, reduce learning rate or check data
4. **Save the TensorBoard files** - they're in `output_mac/runs/`
5. **Take screenshots** of final graphs for documentation

---

## 🎯 Quick Commands

**Start training with monitoring**:
```bash
# Terminal 1: Start training
export HF_TOKEN="your_token"
python tools/finetune_mac.py

# Terminal 2: Start TensorBoard
tensorboard --logdir output_mac/runs
```

**View metrics**: Open http://localhost:6006

**Stop TensorBoard**: Ctrl+C in terminal 2

---

## 📸 Example Output

After training, you'll have graphs showing:
- **Training progress** over time
- **Loss curves** (smooth decrease)
- **Learning rate schedule** (linear decay)
- **Gradient stability** (consistent norm)

Exactly like the Comet ML image, but in TensorBoard!

---

## ✨ Bonus: Save Your Graphs

In TensorBoard:
1. Click the "download" icon on each graph
2. Save as PNG/SVG
3. Include in reports or documentation

---

**Ready to train and monitor?**
```bash
export HF_TOKEN="your_token"
python tools/finetune_mac.py
```

Then in another terminal:
```bash
tensorboard --logdir output_mac/runs
```

Open http://localhost:6006 and watch your model train! 📊✨
