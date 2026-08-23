# Reference images for Klein style steering

Pass **one** style image to the Klein scripts. The experiment folder is named
after the image stem (for example `vangogh.jpg` → `vangogh`).

```bash
bash scripts/run_klein_style.sh data/reference_images/vangogh.jpg
```

That writes:

```text
experiments/klein_9b/style/vangogh/
  reference.jpg
  data_vectors/          # pos / neg activations
  dataset_images/        # extract grids
  final_steering/block_steering/   # *_diff.pt and SVM files
```

Apply later (T2I, no reference image at generate time):

```bash
bash scripts/apply_steering_klein.sh data/reference_images/vangogh.jpg
# or
bash scripts/apply_steering_klein.sh vangogh
```
