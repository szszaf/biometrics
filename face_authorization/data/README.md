# `data/` layout

| Path | In git | Purpose |
|------|--------|---------|
| `celeba_metadata/` | small text files | CelebA identity map (seed) |
| `split/` | `*_split.txt` | Train/valid/test ID lists (seed) |
| `img_align_celeba_cropped/cropped/` | only `.gitkeep` | Cropped 112×112 JPGs — **local only**, huge |
| `system/` | **`.gitignore`** (`*` + `!.gitignore`) | SQLite + WAL — ignored inside the dir; DB created at runtime (`enrollments.db`). If Docker left the folder as **root** and `git pull` cannot create `.gitignore` here, run once: `sudo chown -R "$(id -un):$(id -gn)" face_authorization/data/system` |

Weights live in `../results/` (`.pth` ignored). See repo `README.md`.
