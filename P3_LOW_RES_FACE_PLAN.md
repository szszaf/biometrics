# P3: odporność na zdjęcia twarzy o bardzo niskiej jakości bez fine-tuningu

## Jak to będzie działać

Obraz z kamery lub pliku najpierw trafia do modułu oceny jakości, który sprawdza podstawowe cechy próbki: rozdzielczość, rozmycie, jasność, kontrast oraz to, czy da się wykryć i wyrównać twarz. Jeśli próbka wygląda poprawnie, system używa obecnej ścieżki ArcFace bez dodatkowych zmian.

Jeśli obraz jest słaby, ale nadal używalny, system uruchamia tryb odporny na low-res/CCTV: przygotowuje kilka lekko poprawionych wariantów obrazu, np. po upscalingu, poprawie kontrastu, odszumieniu i wyostrzeniu. Dla każdego wariantu liczony jest embedding twarzy, a następnie embeddingi są uśredniane i normalizowane, aby decyzja nie zależała od pojedynczej niestabilnej wersji obrazu.

Na końcu system porównuje uzyskany embedding z bazą tak jak dotychczas: przez cosine similarity i próg decyzyjny. UI pokazuje wynik oraz informację o jakości próbki: czy użyto standardowej ścieżki, trybu low-quality robust, czy próbka została odrzucona jako zbyt słaba do bezpiecznego uwierzytelnienia.

## Cel

Wdrożyć wariant uwierzytelniania twarzą odporniejszy na zdjęcia o bardzo niskiej rozdzielczości lub jakości, np. CCTV / TinyFace, bez zmiany wag modelu ArcFace. System ma nadal używać obecnego toru:

`MediaPipe Face Landmarker -> alignment 112x112 -> ArcFace -> cosine similarity -> decyzja`

Modyfikacje mają dotyczyć preprocessingu, oceny jakości, sposobu liczenia embeddingu oraz UI/UX. Celem eksperymentalnym jest ograniczenie pogorszenia FRR względem próbek czystych do maksymalnie około 5 punktów procentowych przy możliwie niskim FAR.

## Założenia

- Bez fine-tuningu i bez zmiany pliku `results/arcface_celeba_best.pth`.
- Bez wprowadzania frontendu typu React/Svelte; UI pozostaje w `biometric_authorization/app/static/`.
- Dane surowe nie są zapisywane w bazie systemowej; baza przechowuje embeddingi.
- Próbki bardzo słabe mogą zostać odrzucone z powodem jakościowym zamiast wymuszać niepewną decyzję biometryczną.
- TinyFace traktujemy jako realny zbiór trudnych próbek, a syntetyczne degradacje CelebA jako kontrolowany benchmark do kalibracji progów i raportu.

## Zakres implementacji

### 1. Moduł oceny jakości obrazu

Dodać `biometric_authorization/face_auth/quality.py`.

Planowane elementy:

- `FaceQualityReport` jako Pydantic/dataclass-like struktura wyniku:
  - `width`, `height`,
  - `face_aligned`: czy udało się wykryć i wyrównać twarz,
  - `blur_score`: wariancja Laplacianu,
  - `brightness_mean`,
  - `contrast_std`,
  - `estimated_quality`: `clean`, `low_quality`, `reject`,
  - `warnings`: lista krótkich kodów, np. `low_resolution`, `blurred`, `dark`, `low_contrast`.
- Funkcja `assess_face_image_quality(pil_image: Image.Image, aligned_image: Image.Image | None) -> FaceQualityReport`.
- Progi jako stałe w `face_auth/config.py`, np. minimalny rozmiar wejścia, minimalny kontrast i minimalny blur score.

Ocena jakości nie powinna decydować o tożsamości. Jej rola to:

- wybrać tryb preprocessingu,
- zwrócić ostrzeżenia do UI,
- odrzucić próbkę tylko wtedy, gdy twarz jest niewykrywalna albo jakość jest poniżej minimalnego poziomu bezpieczeństwa.

### 2. Preprocessing dla low-res/CCTV

Dodać `biometric_authorization/face_auth/low_quality.py`.

Planowane funkcje:

- `enhance_low_quality_face(pil_image: Image.Image) -> Image.Image`
  - konwersja do RGB,
  - upscale przez `Image.Resampling.LANCZOS` lub bicubic,
  - delikatne odszumianie,
  - CLAHE na luminancji,
  - lekkie wyostrzenie.
- `make_low_quality_variants(pil_image: Image.Image) -> list[Image.Image]`
  - wariant oryginalny,
  - wariant CLAHE,
  - wariant sharpen,
  - wariant denoise + CLAHE,
  - opcjonalnie wariant z inną interpolacją.

Implementacja powinna użyć zależności już obecnych w projekcie: `PIL`, `numpy`, `cv2`. Nie dodawać ciężkich modeli super-resolution jako domyślnej ścieżki.

### 3. Embedding odporniejszy na niską jakość

Rozszerzyć `biometric_authorization/face_auth/inference.py`.

Nowe funkcje:

- `quality_aware_embedding_from_pil(...)`
  - próbuje standardowego alignmentu,
  - ocenia jakość,
  - dla `clean` używa obecnego `embedding_from_pil`,
  - dla `low_quality` generuje kilka wariantów obrazu, liczy embeddingi i uśrednia je po normalizacji L2,
  - dla `reject` zgłasza kontrolowany błąd z komunikatem jakościowym.
- `quality_aware_embedding_from_bytes(...)`
  - odpowiednik dla endpointów FastAPI.

Ważne: uśrednianie embeddingów ma działać podobnie jak obecne `average_embedding_from_bytes_list`, ale na wariantach jednej próbki zamiast wielu klatek rejestracyjnych.

### 4. Backend API

Rozszerzyć `biometric_authorization/app/main.py`.

Zmiany:

- Dodać modele odpowiedzi:
  - `FaceQualityResponse`,
  - opcjonalnie `FaceAuthQualityMixin` albo pola jakościowe bezpośrednio w odpowiedziach twarzy.
- Dodać endpoint diagnostyczny:
  - `POST /face/quality`
  - wejście: `image`,
  - wyjście: raport jakości bez decyzji biometrycznej.
- Rozszerzyć odpowiedzi dla `POST /verify`, `POST /identify` i `POST /compare` przy `modality=face`:
  - `quality`: raport jakości,
  - `preprocessing_mode`: `standard` albo `low_quality_robust`,
  - `quality_warnings`: krótkie ostrzeżenia dla UI.
- W ścieżkach twarzy zastąpić `embedding_from_bytes(...)` przez `quality_aware_embedding_from_bytes(...)`.
- Nie zmieniać ścieżek głosu.

Decyzja progowa:

- Domyślnie nie obniżać progu dla próbek low-quality.
- Dodać opcjonalny query param `quality_mode=auto|standard|robust`, domyślnie `auto`.
- Próg dla low-quality kalibrować w eksperymentach, ale w produkcyjnym UI nadal pokazywać operatorowi jeden główny próg i informację, że próbka była trudna.

### 5. Eksperymenty i benchmark

Dodać skrypt lub notebook raportowy, np. `biometric_authorization/experiments_low_res_face.ipynb` albo skrypt `biometric_authorization/scripts/evaluate_low_res_face.py`.

Scenariusze danych:

- `clean`: czyste próbki CelebA/test lub obecny split.
- `synthetic_low_res`: kontrolowane degradacje czystych zdjęć:
  - downscale do `8x8`, `12x12`, `16x16`, `24x24`, `32x32`,
  - upscale do `112x112`,
  - JPEG compression,
  - Gaussian blur,
  - noise,
  - zmiana jasności/kontrastu.
- `tinyface`: próbki TinyFace, jeśli uda się zmapować tożsamości i utworzyć pary genuine/impostor.

Metryki:

- dla weryfikacji: FAR, FRR, EER, ROC,
- dla identyfikacji: True Identification Rate, liczba odrzuceń,
- osobno wyniki dla `standard` i `low_quality_robust`,
- osobno wyniki dla poziomów degradacji.

Minimalny raport testowy zgodny z P3:

- co najmniej 1000 próbek czystych,
- co najmniej 5000 próbek trudnych,
- co najmniej 100 zarejestrowanych użytkowników,
- porównanie czyste vs trudne,
- opis metod poprawy odporności bez fine-tuningu.

## UI/UX

### Autoryzacja

Zmodyfikować `biometric_authorization/app/static/index.html`, `app.js` i `styles.css`.

Zmiany w widoku `Autoryzacja`:

- Dodać panel jakości obrazu obok statusu:
  - status: `Dobra jakość`, `Niska jakość`, `Próbka odrzucona`,
  - lista krótkich powodów: `niska rozdzielczość`, `rozmycie`, `ciemny obraz`, `niski kontrast`,
  - informacja, czy użyto trybu `standard` czy `low_quality_robust`.
- Przy skanie twarzy rozbudować kroki:
  - `Klatka zapisana`,
  - `Ocena jakości obrazu`,
  - `Poprawa próbki low-quality` albo `Standardowe przetwarzanie`,
  - `Dopasowanie z bazą`.
- Gdy próbka jest niskiej jakości, wynik nie powinien brzmieć jak pełna pewność. Proponowane komunikaty:
  - pozytywny: `Dostęp przyznany, ale próbka była niskiej jakości. Wynik uzyskano w trybie odpornym na low-res.`
  - negatywny: `Odrzucono. Próbka była niskiej jakości; spróbuj podejść bliżej kamery lub poprawić oświetlenie.`
  - odrzucona jakościowo: `Nie można bezpiecznie uwierzytelnić. Twarz jest zbyt mała lub obraz zbyt rozmyty.`
- Nie pokazywać szczegółów, które pomagają atakującemu nadmiernie optymalizować próbkę; szczegółowe parametry mogą być widoczne tylko w widoku operatora/eksperymentów.

### Rejestracja

Zmiany w widoku `Rejestracja`:

- Przy każdej dodanej klatce pokazać mały znacznik jakości:
  - `OK`,
  - `słaba`,
  - `odrzucona`.
- Blokować zapis profilu, jeśli mniej niż 3 klatki mają akceptowalną jakość.
- Dodać poradę nad kamerą:
  - `Zbierz kilka wyraźnych ujęć. System może przyjąć słabsze próbki przy logowaniu, ale profil referencyjny powinien być możliwie czysty.`
- Zachować obecny model 3-12 klatek i uśrednianie embeddingu.

### Eksperymenty

Zmiany w widoku `Eksperymenty`:

- Dodać sekcję `Low-res / CCTV`:
  - formularz porównania obrazu w trybie standard vs robust,
  - wynik jakości,
  - similarity standard,
  - similarity robust,
  - różnica wyniku,
  - użyty próg.
- Dodać placeholder na wykresy:
  - FRR/FAR dla clean,
  - FRR/FAR dla low-res,
  - FRR wg rozdzielczości twarzy.

### Dostępność i bezpieczeństwo UX

- Wszystkie komunikaty jakości muszą być w `aria-live="polite"`.
- Nie opierać statusu wyłącznie na kolorze; używać tekstu i etykiet.
- Zachować widoczny focus dla nowych przycisków i pól.
- Komunikaty błędów mają być konkretne i możliwe do wykonania, ale bez zdradzania szczegółów wewnętrznej detekcji.
- Nie podpowiadać użytkownikowi dokładnego progu decyzyjnego w komunikatach autoryzacyjnych; próg może pozostać w panelu operatora/eksperymentów.

## Kolejność prac

1. Dodać `face_auth/quality.py` i testy jednostkowe dla oceny blur/jasności/kontrastu.
2. Dodać `face_auth/low_quality.py` i testy sanity-check dla wariantów obrazu.
3. Rozszerzyć `face_auth/inference.py` o `quality_aware_embedding_from_bytes`.
4. Podpiąć nową ścieżkę tylko w endpointach twarzy w `app/main.py`.
5. Dodać `POST /face/quality`.
6. Rozszerzyć odpowiedzi API o pola jakościowe.
7. Zaktualizować UI autoryzacji: panel jakości i komunikaty.
8. Zaktualizować UI rejestracji: jakość klatek i blokada słabych profili.
9. Zaktualizować UI eksperymentów: standard vs robust.
10. Przygotować skrypt/notebook ewaluacyjny dla clean, synthetic low-res i TinyFace.
11. Skalibrować progi jakości i próg decyzyjny na walidacji.
12. Uzupełnić raport P3 wynikami i wnioskami.

## Kryteria akceptacji

- Standardowa ścieżka twarzy nadal działa dla czystych próbek.
- Próbki low-res przechodzą przez tryb robust bez fine-tuningu modelu.
- API zwraca informację o jakości próbki i użytym trybie preprocessingu.
- UI jasno pokazuje, kiedy próbka była niskiej jakości i co użytkownik może poprawić.
- System potrafi odrzucić próbkę zbyt słabą bez fałszywej pewności.
- Eksperymenty pozwalają porównać clean vs low-res na wymaganych liczbach próbek.

## Ryzyka

- Super-resolution lub zbyt agresywne wyostrzanie może zmienić cechy twarzy i zwiększyć FAR; dlatego domyślnie stosujemy lekkie przetwarzanie i uśrednianie embeddingów.
- Obniżanie progu dla low-quality może poprawić FRR kosztem FAR; próg należy dobierać na walidacji.
- TinyFace może być trudny do użycia w weryfikacji 1:1, jeśli mapowanie tożsamości i par genuine/impostor będzie ograniczone. W takim przypadku głównym benchmarkiem powinny być kontrolowane degradacje CelebA, a TinyFace może być dodatkowym testem jakościowym.
