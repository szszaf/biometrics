# `data/` layout

Zawartość tego katalogu **nie jest wersjonowana** (`.gitignore`: `biometric_authorization/data/*` z wyjątkiem tego pliku). Przygotuj strukturę lokalnie lub przez Docker (montowanie wolumenu).

| Path | W repozytorium | Purpose |
|------|----------------|---------|
| `README.md` | tak | Ten plik — opis układu |
| `celeba_metadata/` | nie | CelebA identity map (seed twarzy) |
| `split/` | nie | Listy ID train/valid/test (seed twarzy) |
| `voice_split/` | nie | Listy mówców CV PL (seed głosu) |
| `img_align_celeba_cropped/cropped/` | nie | Cropy 112×112 JPG — duże, tylko lokalnie |
| `cv-corpus-*/pl/` | nie | Common Voice PL po rozpakowaniu archiwum |
| `system/` | nie | SQLite + WAL (`enrollments*.db`) — tworzone w runtime |

Wagi modeli: `../results/` (`.pth` w `.gitignore` w korzeniu projektu). Szczegóły uruchomienia: główne `README.md` repozytorium.

Jeśli Docker utworzył pliki jako **root** i `git` nie może zapisać w `data/system/`, jednorazowo:  
`sudo chown -R "$(id -un):$(id -gn)" biometric_authorization/data/system`
