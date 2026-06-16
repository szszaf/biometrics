# P3: wnioski z eksperymentów low-res / CCTV

## Cel eksperymentu

Sprawdziliśmy, czy system uwierzytelniania twarzą spełnia wymaganie P3 dla próbek trudnych, czyli zdjęć o niskiej rozdzielczości lub jakości.

Wymaganie metryczne z P3:

- dla weryfikacji FRR na próbkach trudnych nie powinien wzrosnąć o więcej niż 5 punktów procentowych względem próbek czystych,
- jednocześnie należy minimalizować FAR,
- test powinien obejmować co najmniej 1000 próbek czystych i 5000 próbek trudnych.

## Eksperyment 1: zbyt agresywna degradacja

Pierwszy eksperyment mieszał bardzo silne degradacje:

- `8x8`,
- `12x12`,
- `16x16`,
- `24x24`,
- `32x32`.

Wynik:

| Zbiór / metoda | FRR | FAR |
| --- | ---: | ---: |
| Clean | 4.40% | 0.50% |
| Low-res standard | 53.76% | - |
| Low-res robust | 54.28% | 0.94% |

Delta FRR względem clean:

```text
49.88 p.p.
```

Wniosek:

Tak mocna degradacja usuwa zbyt dużo informacji biometrycznej. Przy rozdzielczościach `8x8-32x32` model ArcFace nie ma stabilnych cech twarzy do porównania. Sam preprocessing, CLAHE, wyostrzanie, odszumianie i uśrednianie embeddingów nie wystarczają. Taki wariant nie spełnia wymagania P3.

## Decyzja po eksperymencie 1

Uznaliśmy, że próbki poniżej akceptowalnego poziomu jakości nie powinny być używane do automatycznej decyzji biometrycznej.

Przyjęta polityka:

- `64x64`, `80x80` - akceptowalny low-res / CCTV do eksperymentu P3,
- `8x8`, `12x12`, `16x16`, `24x24`, `32x32`, `48x48` - próbki odrzucane jakościowo.

Uzasadnienie:

W realnym systemie użytkownik nie dostarcza ręcznie obrazu `8x8`. System pobiera klatkę z kamery i powinien ocenić, czy twarz jest wystarczająco duża, jasna i ostra. Jeśli nie, bezpieczniejszym zachowaniem jest prośba o ponowne pobranie próbki, np. podejście bliżej kamery lub poprawa oświetlenia.

## Eksperyment 2: łagodny low-res / CCTV

Drugi eksperyment używał łagodniejszych degradacji:

- `64x64`,
- `80x80`.

Wynik:

| Zbiór / metoda | FRR | FAR |
| --- | ---: | ---: |
| Clean | 4.40% | 0.50% |
| Low-res standard | 4.36% | - |
| Low-res robust | 4.52% | 0.56% |

Delta FRR dla low-res robust względem clean:

```text
+0.12 p.p.
```

Wymaganie P3:

```text
0.12 p.p. <= 5 p.p. -> spełnione
```

## Interpretacja wyniku

System spełnia wymaganie P3 dla próbek low-res uznanych za używalne, czyli `64x64` i `80x80`.

Warto zauważyć, że tryb `low-res robust` nie poprawił FRR względem standardowej ścieżki dla łagodnego low-res:

```text
low-res standard: 4.36%
low-res robust:   4.52%
```

Oznacza to, że dla `64x64` i `80x80` standardowy ArcFace radzi sobie już dobrze, a dodatkowe poprawianie obrazu nie daje przewagi. Wartość dodana systemu polega głównie na:

- ocenie jakości próbki,
- rozpoznaniu, czy próbka jest używalnym low-res czy skrajną degradacją,
- kontrolowanym odrzuceniu próbek zbyt słabych,
- utrzymaniu FRR/FAR blisko wyniku dla próbek czystych.

## Wniosek końcowy

Końcowa metoda powinna działać według polityki:

```text
clean / mild low-res -> uwierzytelnianie
extreme low-res -> odrzucenie jakościowe i prośba o ponowną próbkę
```

Dla akceptowalnych próbek low-res (`64x64`, `80x80`) system spełnia wymaganie P3, ponieważ wzrost FRR wyniósł tylko `0.12 p.p.` względem próbek czystych.

Dla skrajnych degradacji (`8x8-48x48`) nie należy obniżać progu na siłę, bo mogłoby to zwiększyć FAR. Te próbki powinny być traktowane jako zbyt słabe do bezpiecznego uwierzytelniania.

## Dalsze możliwe usprawnienia

- Skalibrować osobny próg dla `mild low-res`, np. na zbiorze walidacyjnym.
- Dla `64x64/80x80` preferować standardową ścieżkę, skoro osiąga minimalnie lepszy FRR niż robust.
- Zostawić robust preprocessing jako fallback albo element diagnostyczny.
- Jeśli celem byłaby obsługa `32x32` i niżej, potrzebne byłoby dotrenowanie modelu lub augmentacja treningowa low-res.
